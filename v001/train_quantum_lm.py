# train_quantum_lm.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from pathlib import Path

from quantum_lm import QuantumLM, VOCAB_SIZE, CONTEXT_LEN, DEVICE


# Fraction of training lines to sample per epoch for faster debugging
SAMPLE_FRACTION = 1.0


# -----------------------------
# Example dataset stub
# -----------------------------

class TinyStoriesDataset(Dataset):
    """
    Dataset backed directly by a token-id text file.

    Each line in the file is a space-separated list of integer token IDs
    representing one story or chunk.

    For each __getitem__ call, we:
      - read the corresponding line,
      - parse the token IDs,
      - choose a random prediction position t in [1, len(ids) - 1],
      - take up to `context_len` tokens before t as the context,
      - and use ids[t] as the target.

    This avoids materializing all sliding windows in memory.
    """
    def __init__(self, ids_path: str, context_len: int = CONTEXT_LEN):
        self.ids_path = ids_path
        self.context_len = context_len
        self.offsets = []

        # Build line offsets so we can seek to each story line without
        # keeping all token IDs in memory.
        with open(self.ids_path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                self.offsets.append(pos)

        if not self.offsets:
            raise RuntimeError(f"No non-empty lines found in {self.ids_path}")

    def __len__(self) -> int:
        # One sample per line per epoch; each __getitem__ picks a random slice.
        return len(self.offsets)

    def __getitem__(self, idx: int):
        offset = self.offsets[idx]
        with open(self.ids_path, "rb") as f:
            f.seek(offset)
            line = f.readline().decode("utf-8").strip()

        if not line:
            # Shouldn't normally happen because we skip empty lines in __init__
            raise IndexError(f"Empty line at index {idx}")

        ids = [int(tok) for tok in line.split() if tok.strip()]

        if len(ids) < 2:
            # Not enough tokens to form (context, target); skip by wrapping around.
            # In practice, TinyStories lines should be much longer than this.
            return self.__getitem__((idx + 1) % len(self))

        # Choose a random prediction position t in [1, len(ids) - 1]
        t = torch.randint(1, len(ids), (1,)).item()

        start = max(0, t - self.context_len)
        ctx_tokens = ids[start:t]

        # Left-pad context if shorter than context_len
        if len(ctx_tokens) < self.context_len:
            pad_len = self.context_len - len(ctx_tokens)
            ctx_tokens = [0] * pad_len + ctx_tokens  # 0 is [UNK] in our vocab

        context = torch.tensor(ctx_tokens, dtype=torch.long)
        target = int(ids[t])

        return context, target


def collate_batch(batch):
    """
    batch: list of (context, target) pairs.
    We stack contexts; all are already length CONTEXT_LEN after padding.
    """
    contexts, targets = zip(*batch)
    contexts = torch.stack(contexts, dim=0)     # (B, T)
    targets = torch.tensor(targets, dtype=torch.long)  # (B,)
    return contexts, targets


# -----------------------------
# Training loop
# -----------------------------

def train_epoch(model, dataloader, optimizer, epoch: int):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for step, (contexts, targets) in enumerate(dataloader):
        contexts = contexts.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        log_probs = model(contexts)          # (B, V)
        loss = nn.NLLLoss()(log_probs, targets)

        loss.backward()
        optimizer.step()

        batch_size = contexts.size(0)
        total_loss += loss.item() * batch_size
        total_tokens += batch_size

        print(f"Epoch {epoch} | Step {step+1}/{len(dataloader)}", end="\r")
        if (step + 1) % 100 == 0:
            print(f"Epoch {epoch} | Step {step+1}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(1, total_tokens)
    print(f"Epoch {epoch} completed. Avg loss: {avg_loss:.4f}")


def main():
    # Load token-id sequences from preprocessed TinyStories file (streaming).
    #train_ids_path = Path("tinystories/train_ids.txt")
    train_ids_path = Path("tinier-stories/entropica_1024.combined.clean.txt")
    if not train_ids_path.exists():
        raise RuntimeError(f"Could not find {train_ids_path}. "
                           f"Run your preprocessing script to generate it.")

    print(f"Indexing training sequences from {train_ids_path} ...")
    dataset = TinyStoriesDataset(str(train_ids_path), context_len=CONTEXT_LEN)
    print(f"Dataset initialized with {len(dataset)} lines "
          f"(one random context-target pair per line per epoch).")

    # Build a sampler that uses only a random fraction of the dataset each epoch
    num_lines = len(dataset)
    sample_size = max(1, int(num_lines * SAMPLE_FRACTION))
    print(f"Sampling {sample_size} lines (~{SAMPLE_FRACTION*100:.1f}% of {num_lines}) "
          f"per epoch for faster debugging.")
    # Random subset of indices
    subset_indices = torch.randperm(num_lines)[:sample_size].tolist()
    sampler = torch.utils.data.SubsetRandomSampler(subset_indices)

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        collate_fn=collate_batch,
        num_workers=0,
    )

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    model = QuantumLM(vocab_size=VOCAB_SIZE).to(DEVICE)
    #model = torch.compile(model) # one time performance hit
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Save a special checkpoint capturing the randomly initialized model state
    random_ckpt_path = ckpt_dir / "random_start.pt"
    torch.save(
        {
            "epoch": 0,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        random_ckpt_path,
    )
    print(f"Saved random-start checkpoint to {random_ckpt_path}")

    num_epochs = 500
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch: {epoch}")
        train_epoch(model, dataloader, optimizer, epoch)

        # Save checkpoints (full and lightweight latest)
        ckpt_path = ckpt_dir / f"quantum_lm_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            ckpt_path,
        )
        torch.save(model.state_dict(), ckpt_dir / "quantum_lm_latest.pt")
        print(f"Saved checkpoint to {ckpt_path}")

    # Save final model (for convenience)
    torch.save(model.state_dict(), "quantum_lm_tinystories.pt")
    print("Training complete, model saved.")


if __name__ == "__main__":
    main()
