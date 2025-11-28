# sample_quantum_lm.py

import sys
from pathlib import Path
import re

import torch

from quantum_lm import QuantumLM, VOCAB_SIZE, CONTEXT_LEN, DEVICE

UNK_TOKEN = "[UNK]"
VOCAB_PATH = Path("tinystories/vocab.txt")
CKPT_DIR = Path("checkpoints")
VALID_WORD_RE = re.compile(r"^[a-z]+$", re.IGNORECASE)

ENTROPICA_1024_VOCAB = [
    "[EOS]", "the", "and", "to", "a", "was", "he", "she", "they", "it", "her", "said", "in", "you", "his",
    "with", "that", "so", "of", "day", "but", "had", "on", "one", "for", "there", "very", "little", "time",
    "i", "happy", "is", "not", "big", "saw", "once", "were", "play", "are", "wanted", "upon", "their",
    "him", "girl", "them", "be", "have", "can", "at", "up", "go", "went", "all", "friends", "when", "we",
    "from", "what", "then", "back", "smiled", "mom", "dad", "home", "out", "like", "good", "new", "some",
    "fun", "made", "an", "my", "me", "came", "no", "too", "now", "will", "into", "about", "your", "do",
    "if", "just", "as", "by", "has", "more", "would", "could", "did", "who", "which", "or", "been", "its",
    "how", "than", "after", "over", "two", "first", "other", "way", "off", "old", "because", "any", "much",
    "only", "see", "even", "must", "own", "us", "our", "let", "know", "take", "where", "am", "may", "well",
    "down", "should", "each", "most", "people", "mr", "get", "work", "life", "before", "great", "same",
    "right", "mean", "use", "still", "also", "many", "such", "through", "long", "being", "might", "say",
    "again", "think", "made", "why", "ask", "men", "never", "here", "car", "give", "world", "last", "need",
    "feel", "try", "few", "something", "around", "another", "house", "while", "found", "come", "part",
    "place", "point", "help", "put", "end", "does", "name", "away", "hand", "show", "every", "tell",
    "small", "set", "three", "want", "air", "got", "thought", "mother", "father", "left", "boy", "dog",
    "cat", "friend", "love", "tree", "ball", "red", "blue", "green", "yellow", "bird", "fish", "ran",
    "jumped", "played", "laughed", "cried", "walked", "home", "school", "park", "cake", "ice", "cream",
    "sun", "moon", "star", "flower", "rain", "snow", "hot", "cold", "fast", "slow", "yes", "no", "please",
    "thank", "you", "hello", "goodbye", "morning", "night", "eat", "drink", "sleep", "wake", "run", "jump",
    "sing", "dance", "read", "write", "draw", "paint", "make", "find", "look", "listen", "talk", "hug",
    "kiss", "smile", "sad", "mad", "scared", "brave", "kind", "nice", "mean", "share", "help", "wait",
    "turn", "win", "lose", "try", "best", "pretty", "ugly", "loud", "quiet", "clean", "dirty", "wet", "dry",
    "open", "close", "high", "low", "near", "far", "left", "right", "front", "back", "top", "bottom",
    "inside", "outside", "under", "over", "next", "between", "behind", "together", "alone", "always",
    "sometimes", "never", "today", "tomorrow", "yesterday", "now", "later", "soon", "already", "still",
    "again", "once", "twice", "many", "few", "all", "none", "some", "any", "more", "less", "most", "least",
    "big", "bigger", "biggest", "small", "smaller", "smallest", "good", "better", "best", "bad", "worse",
    "worst", "happy", "happier", "happiest", "sad", "sadder", "saddest", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine", "ten", "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "hers", "ours",
    "theirs", "this", "that", "these", "those", "who", "what", "where", "when", "why", "how", "which",
    "whose", "a", "an", "the", "and", "but", "or", "nor", "for", "yet", "so", "after", "although",
    "because", "before", "if", "once", "since", "though", "unless", "until", "when", "while", "as",
    "by", "in", "of", "on", "to", "with", "at", "from", "into", "onto", "out", "over", "under", "up",
    "down", "off", "on", "again", "away", "back", "forward", "here", "there", "everywhere", "nowhere",
    "somewhere", "anywhere", "home", "school", "park", "store", "city", "country", "world", "sun", "moon",
    "star", "sky", "cloud", "rain", "snow", "wind", "flower", "tree", "grass", "river", "lake", "sea",
    "beach", "mountain", "hill", "valley", "forest", "desert", "island", "road", "path", "street", "house",
    "room", "door", "window", "bed", "chair", "table", "book", "toy", "game", "ball", "doll", "car", "bike",
    "train", "plane", "boat", "ship", "dog", "cat", "bird", "fish", "rabbit", "mouse", "bear", "lion",
    "tiger", "elephant", "monkey", "giraffe", "zebra", "horse", "cow", "pig", "chicken", "duck", "frog",
    "snake", "spider", "bee", "butterfly", "ant", "ladybug", "worm", "squirrel", "deer", "fox", "wolf",
    "mom", "dad", "brother", "sister", "grandma", "grandpa", "uncle", "aunt", "cousin", "baby", "child",
    "friend", "teacher", "doctor", "police", "fireman", "mailman", "farmer", "cook", "artist", "singer",
    "dancer", "player", "hero", "princess", "king", "queen", "prince", "dragon", "castle", "magic", "wish",
    "dream", "love", "hate", "like", "want", "need", "feel", "think", "know", "see", "hear", "touch",
    "taste", "smell", "eat", "drink", "sleep", "wake", "walk", "run", "jump", "swim", "fly", "climb",
    "fall", "sit", "stand", "lie", "laugh", "cry", "smile", "frown", "shout", "whisper", "sing", "dance",
    "read", "write", "draw", "paint", "build", "break", "fix", "clean", "dirty", "open", "close", "start",
    "stop", "begin", "end", "win", "lose", "try", "fail", "succeed", "help", "hurt", "share", "take",
    "give", "receive", "buy", "sell", "find", "lose", "keep", "throw", "catch", "kick", "hit", "hug",
    "kiss", "hold", "carry", "push", "pull", "lift", "drop", "wait", "hurry", "slow", "fast", "early",
    "late", "today", "tomorrow", "yesterday", "morning", "afternoon", "evening", "night", "week", "month",
    "year", "spring", "summer", "fall", "winter", "hot", "cold", "warm", "cool", "wet", "dry", "bright",
    "dark", "light", "heavy", "loud", "quiet", "soft", "hard", "smooth", "rough", "sweet", "sour", "salty",
    "bitter", "yummy", "yucky", "beautiful", "ugly", "pretty", "handsome", "cute", "scary", "funny",
    "serious", "kind", "mean", "nice", "rude", "brave", "scared", "strong", "weak", "smart", "silly",
    "good", "bad", "happy", "sad", "angry", "calm", "excited", "bored", "tired", "awake", "hungry", "full",
    "thirsty", "sick", "healthy", "big", "small", "tall", "short", "long", "short", "wide", "narrow",
    "thick", "thin", "old", "young", "new", "old", "clean", "dirty", "rich", "poor", "safe", "dangerous",
    "easy", "hard", "fun", "boring", "interesting", "simple", "difficult", "right", "wrong", "true",
    "false", "yes", "no", "maybe", "always", "never", "sometimes", "often", "rarely", "all", "none",
    "some", "many", "few", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    # extra common flexible words
    "walked", "walking", "played", "playing", "ran", "running", "jumping", "smiled", "smiling", "laughed",
    "laughing", "cried", "crying", "hugged", "hugging", "kissing", "giving", "gave", "taken", "taking",
    "holding", "held", "opened", "opening", "closed", "closing", "called", "calling", "telling", "told",
    "asking", "asked", "answer", "answered", "answering", "looking", "looked", "watching", "watched",
    "listened", "listening", "heard", "hearing", "feeling", "felt", "thinking", "thought", "knowing",
    "known", "sharing", "shared", "helped", "helping", "stayed", "staying", "sat", "sitting",
    "stood", "standing", "reading", "writing", "drawing", "painting", "building", "built", "fixing",
    "fixed", "cleaning", "brighter", "brightest", "darker", "darkest", "warmer", "warmest", "cooler",
    "coolest", "softer", "softest", "harder", "hardest", "slower", "slowest", "faster", "fastest",
    "earlier", "earliest", "later", "latest", "nearby", "farther", "farthest", "closer", "closest",
    "upstairs", "downstairs", "anytime", "someday", "tonight", "maybe", "really", "quite", "almost",
    "together", "apart", "beside", "between", "behind", "above", "below", "across", "through", "toward",
    "onto", "away", "forward", "middle", "corner", "edge", "center", "circle", "square", "line", "box",
    "bag", "hat", "shirt", "pants", "shoes", "coat", "dress", "blanket", "pillow", "floor", "ceiling",
    "wall", "garden", "yard", "porch", "step", "stairs", "swing", "slide", "sandbox", "tower", "bridge",
    "puddle", "shadow", "noise", "quietly", "softly", "gently", "quickly", "slowly", "suddenly",
    "carefully", "happily", "sadly", "angrily", "bravely", "kindly", "nicely", "safely", "alone",
    "story", "stories", "letter", "letters", "page", "pages", "picture", "pictures", "photo", "photos",
    "map", "maps", "shape", "shapes", "color", "colors", "rock", "rocks", "stone", "stones", "stick",
    "sticks", "leaf", "leaves", "branch", "branches", "pathway", "field", "fields", "farm", "farms",
    "barn", "barns", "market", "markets", "shop", "shops", "kitchen", "bathroom", "hall", "hallway",
    "attic", "basement", "shelf", "shelves", "pocket", "pockets", "window", "windows", "curtain",
    "curtains", "cushion", "cushions", "sofa", "couch", "mirror", "clock", "clocks",
    "balloon", "balloons", "block", "blocks", "puzzle", "puzzles", "rope", "string", "ribbon", "toybox",
    "robot", "robots", "train", "trains", "engine", "engines", "drum", "drums", "guitar", "piano",
    "violin", "flute", "song", "songs", "music", "rhythm", "pattern", "patterns",
    "cookie", "cookies", "candy", "candies", "chocolate", "bread", "butter", "cheese", "soup",
    "sandwich", "sandwiches", "milk", "juice", "water", "tea", "breakfast", "lunch", "dinner", "snack",
    "apple", "apples", "banana", "bananas", "grape", "grapes", "pear", "pears", "peach", "peaches",
    "orange", "oranges", "berry", "berries", "carrot", "carrots", "potato", "potatoes",
    "heroic", "gentle", "worried", "curious", "proud", "lonely", "peaceful", "playful", "friendly",
    "helpful", "honest", "polite", "patient", "lucky", "unlucky", "carefree", "serious", "quiet",
    "shy", "nervous", "surprised", "sleepy", "noisy", "messy", "tidy"
]


# Remove duplicates and limit to 1023 + [EOS]
VOCAB = ["[EOS]"] + list(dict.fromkeys(ENTROPICA_1024_VOCAB[1:]))[:1023]

def load_vocab(vocab_path: Path):
    """
    Load vocab.txt of form: id \t token
    Returns (id_to_token, token_to_id).
    """
    id_to_token = {}
    token_to_id = {}

    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, tok = line.split("\t")
            idx = int(idx_str)
            id_to_token[idx] = tok
            token_to_id[tok] = idx

    # Safety: ensure UNK exists, else map unknowns to 0
    if UNK_TOKEN not in token_to_id:
        token_to_id[UNK_TOKEN] = 0
        if 0 not in id_to_token:
            id_to_token[0] = UNK_TOKEN

    return id_to_token, token_to_id


def normalize_word(word: str) -> str:
    """
    Match the training-time normalization:
    - lowercase
    - any non-alpha token becomes [UNK]
    """
    w = word.lower().strip()
    if not VALID_WORD_RE.match(w):
        return UNK_TOKEN
    return w


def tokenize_prompt(prompt: str):
    """
    Simple tokenizer similar to preprocessing:
    words vs non-word chars, then normalize.
    """
    # split into word-like and non-word chars
    raw_tokens = re.findall(r"[A-Za-z']+|[^A-Za-z\s]", prompt)
    norm = [normalize_word(t) for t in raw_tokens]
    # collapse consecutive UNKs a bit (optional)
    # norm = [t for i, t in enumerate(norm) if not (t == UNK_TOKEN and i > 0 and norm[i-1] == UNK_TOKEN)]
    return norm


def prompt_to_ids(prompt: str, token_to_id: dict):
    toks = tokenize_prompt(prompt)
    ids = [token_to_id.get(t, token_to_id.get(UNK_TOKEN, 0)) for t in toks]
    if not ids:
        ids = [token_to_id.get(UNK_TOKEN, 0)]
    return ids


def load_model_from_latest_checkpoint(ckpt_override: Path | None = None) -> QuantumLM:
    """
    Load QuantumLM and initialize from a checkpoint.

    - If ckpt_override is provided, load exactly that file.
    - Otherwise:
        * Look for checkpoints/quantum_lm_latest.pt (state_dict only),
        * Fall back to last full epoch checkpoint (with 'model_state').
    """
    model = QuantumLM(vocab_size=VOCAB_SIZE).to(DEVICE)

    # If user explicitly requested a checkpoint, use that
    if ckpt_override is not None:
        ckpt_path = ckpt_override
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Requested checkpoint {ckpt_path} does not exist")

        print(f"Loading checkpoint from {ckpt_path} ...")
        obj = torch.load(ckpt_path, map_location=DEVICE)

        # Heuristic: either a raw state_dict or a dict with 'model_state'
        if isinstance(obj, dict) and "model_state" in obj:
            model.load_state_dict(obj["model_state"])
        else:
            model.load_state_dict(obj)

        model.eval()
        return model

    # Default behavior: latest state_dict, then last epoch ckpt
    latest_sd = CKPT_DIR / "quantum_lm_latest.pt"
    if latest_sd.exists():
        print(f"Loading latest state_dict from {latest_sd} ...")
        state_dict = torch.load(latest_sd, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    candidates = sorted(CKPT_DIR.glob("quantum_lm_epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {CKPT_DIR}")

    last_ckpt = candidates[-1]
    print(f"Loading full checkpoint from {last_ckpt} ...")
    ckpt = torch.load(last_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def generate(
    model: QuantumLM,
    initial_ids,
    id_to_token,
    token_to_id,
    max_new_tokens: int = 30,
    top_k: int = 10,
    top_p: float | None = None,
    temperature: float = 1.0,
):
    """
    Autoregressive generation:
    - Start from initial_ids
    - Repeatedly predict next token and append
    """
    seq = list(initial_ids)

    # Cache the EOS token id for later use
    eos_id = token_to_id.get("[EOS]", None)

    for step in range(max_new_tokens):
        # Build context window
        ctx = seq[-CONTEXT_LEN:]
        if len(ctx) < CONTEXT_LEN:
            pad_len = CONTEXT_LEN - len(ctx)
            ctx = [token_to_id.get(UNK_TOKEN, 0)] * pad_len + ctx

        context_tensor = torch.tensor(ctx, dtype=torch.long, device=DEVICE).unsqueeze(0)
        log_probs = model(context_tensor)  # (1, V)

        # apply temperature scaling before softmax
        if temperature != 1.0:
            log_probs = log_probs / temperature

        probs = log_probs.exp().squeeze(0)  # (V,)

        # 🔹 Restrict sampling to the actually defined vocab size
        active_vocab_size = len(id_to_token)  # length of ENTROPICA_1024_VOCAB
        if active_vocab_size < probs.numel():
            # zero out probabilities beyond the last valid token
            probs = probs.clone()
            probs[active_vocab_size:] = 0
            # re-normalize to sum to 1
            total = probs.sum()
            if total > 0:
                probs = probs / total
            else:
                # extremely defensive; fallback to uniform over active vocab
                probs[:active_vocab_size] = 1.0 / active_vocab_size

        # 🔹 Penalize over-use of EOS to avoid [EOS] spam
        if eos_id is not None and eos_id < probs.numel():
            # Before the last few steps, make EOS less likely,
            # and if the last emitted token was EOS, downweight it even more.
            if step < max_new_tokens - 3:
                probs = probs.clone()
                if seq and seq[-1] == eos_id:
                    probs[eos_id] *= 0.05
                else:
                    probs[eos_id] *= 0.2
                total = probs.sum()
                if total > 0:
                    probs = probs / total

        # Nucleus (top-p) or top-k sampling
        if top_p is not None:
            # sort probabilities in descending order
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=0)
            # keep smallest set of tokens with total mass >= top_p
            mask = cumulative <= top_p
            # ensure at least one token is kept
            if not mask.any():
                mask[0] = True
            filtered_probs = sorted_probs * mask
            filtered_probs = filtered_probs / filtered_probs.sum()
            sampled_idx = torch.multinomial(filtered_probs, 1).item()
            next_id = sorted_idx[sampled_idx].item()
        elif top_k is not None and top_k < probs.numel():
            # Top-k sampling to avoid super-flat tails
            top_k_vals, top_k_idx = torch.topk(probs, top_k)
            top_k_probs = top_k_vals / top_k_vals.sum()
            next_id = top_k_idx[torch.multinomial(top_k_probs, 1).item()].item()
        else:
            # full multinomial over all tokens
            probs = probs / probs.sum()
            next_id = torch.multinomial(probs, 1).item()

        seq.append(next_id)

    # Convert back to tokens
    raw_tokens = [id_to_token.get(i, UNK_TOKEN) for i in seq]

    # 🔹 Post-process: treat [EOS] as sentence boundary and build a nicer string
    words = []
    for tok in raw_tokens:
        if tok == "[EOS]":
            # end of sentence: add a period if we have content and don't already end with punctuation
            if words and words[-1] not in [".", "!", "?", ","]:
                words.append(".")
            # don't keep the [EOS] token itself
            continue
        words.append(tok)

    # If we somehow ended without punctuation, add a final period
    if words and words[-1] not in [".", "!", "?"]:
        words.append(".")

    # Join and clean up spacing before punctuation
    text = " ".join(words)
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")

    # Return human-readable text plus raw tokens/ids for inspection
    return text, raw_tokens, seq


def main():

    if len(sys.argv) < 2:
        print("Usage: python3 sample_quantum_lm.py \"your prompt here\" [--temp T] [--top_k K] [--top_p P]")
        sys.exit(1)

    args = sys.argv[1:]

    # default sampling params
    temperature = 0.8
    top_k = 10
    top_p = 0.9

    # optional checkpoint override
    ckpt_path: Path | None = None

    # everything up to the first --flag is treated as the prompt
    prompt_tokens = []
    i = 0
    while i < len(args) and not args[i].startswith("--"):
        prompt_tokens.append(args[i])
        i += 1

    if not prompt_tokens:
        print("Error: you must provide a text prompt before any flags.")
        sys.exit(1)

    # parse optional flags
    while i < len(args):
        if args[i] == "--temp" and i + 1 < len(args):
            temperature = float(args[i + 1])
            i += 2
        elif args[i] == "--top_k" and i + 1 < len(args):
            top_k = int(args[i + 1])
            i += 2
        elif args[i] == "--top_p" and i + 1 < len(args):
            top_p = float(args[i + 1])
            i += 2
        elif args[i] == "--ckpt" and i + 1 < len(args):
            ckpt_path = Path(args[i + 1])
            i += 2
        else:
            print(f"Warning: ignoring unrecognized or incomplete arg: {args[i]}")
            i += 1

    prompt = " ".join(prompt_tokens)
    print(f"Prompt: {prompt}")
    print(f"Sampling params: temperature={temperature}, top_k={top_k}, top_p={top_p}")

    if not CKPT_DIR.exists():
        raise RuntimeError(f"Checkpoint directory {CKPT_DIR} does not exist")

    # Build vocab mappings directly from ENTROPICA_1024_VOCAB defined in this file
    #id_to_token = {i: tok for i, tok in enumerate(ENTROPICA_1024_VOCAB)}
    id_to_token = {i: tok for i, tok in enumerate(ENTROPICA_1024_VOCAB)}
    token_to_id = {tok: i for i, tok in id_to_token.items()}

    # Ensure UNK_TOKEN has a defined id (default to 0 if not present)
    if UNK_TOKEN not in token_to_id:
        token_to_id[UNK_TOKEN] = 0

    initial_ids = prompt_to_ids(prompt, token_to_id)

    print(f"Initial token IDs: {initial_ids}")

    model = load_model_from_latest_checkpoint(ckpt_override=ckpt_path)

    # Top-k sample - OG
    generated_topk, tokens_topk, ids_topk = generate(
        model,
        initial_ids,
        id_to_token,
        token_to_id,
        max_new_tokens=30,
        top_k=top_k,
        top_p=None,
        temperature=1.0,
    )

    print("\n=== Top-k sample (k={}, T={}) ===".format(10, 1.0))
    print(generated_topk)
    print("=================")

    this_temp = 0.7
    this_top_k = 5
    # Top-k sample
    generated_topk, tokens_topk, ids_topk = generate(
        model,
        initial_ids,
        id_to_token,
        token_to_id,
        max_new_tokens=30,
        top_k=this_top_k,
        top_p=None,
        temperature=this_temp,
    )

    print("\n=== Top-k sample (k={}, T={}) ===".format(this_top_k, this_temp))
    print(generated_topk)
    print("=================")

    # Nucleus (top-p) sample
    generated_topp, tokens_topp, ids_topp = generate(
        model,
        initial_ids,
        id_to_token,
        token_to_id,
        max_new_tokens=30,
        top_k=None,
        top_p=top_p,
        temperature=temperature,
    )


    print("\n=== Nucleus (top-p) sample (p={}, T={}) ===".format(top_p, temperature))
    print(generated_topp)
    print("=================")


if __name__ == "__main__":
    main()
