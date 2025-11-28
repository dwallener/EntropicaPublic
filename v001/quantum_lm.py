# quantum_lm.py

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Hyperparameters
# -----------------------------

VOCAB_SIZE = 1024       # including [UNK], and you can reserve a few for [PAD], [BOS], [EOS]
EMBED_DIM = 512         # real-valued embedding dimension
NUM_MODES = 1024         # D: dimension of optical Hilbert space (number of modes)
CONTEXT_LEN = 128       # max context length
NUM_MZI_LAYERS = 32      # number of mesh layers (depth of quantum core)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Utility: complex normalization & softmax
# -----------------------------

def complex_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize a complex vector along last dim.
    x: (..., D) complex tensor
    """
    # |x|^2 = x* conj(x)
    mag_sq = (x.real ** 2 + x.imag ** 2).sum(dim=-1, keepdim=True)
    mag = torch.sqrt(mag_sq + eps)
    return x / mag


def born_log_probs(amps: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Convert complex amplitudes to log-probabilities using the Born rule:
    p_i = |a_i|^2 / sum_j |a_j|^2

    amps: (B, V) complex tensor
    returns: (B, V) log-probs
    """
    mag_sq = amps.real ** 2 + amps.imag ** 2  # (B, V), real >= 0
    mag_sq = mag_sq + eps
    log_mag_sq = torch.log(mag_sq)           # log |a_i|^2

    # log p_i = log |a_i|^2 - log sum_j |a_j|^2
    log_norm = torch.logsumexp(log_mag_sq, dim=-1, keepdim=True)
    log_probs = log_mag_sq - log_norm
    return log_probs


# -----------------------------
# Context encoder (digital)
# -----------------------------

class ContextEncoder(nn.Module):
    """
    Digital encoder: takes integer tokens (B, T) and outputs
    a complex initial state psi0 in C^NUM_MODES.

    We:
      - embed tokens into R^EMBED_DIM
      - run a GRU (or just average)
      - project to R^(2*NUM_MODES), then interpret as complex (real, imag)
      - normalize to unit norm
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_modes: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.to_complex = nn.Linear(embed_dim, 2 * num_modes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, T) int64
        returns: psi0: (B, D) complex
        """
        x = self.embed(tokens)      # (B, T, E)
        # Simple GRU over time
        _, h_n = self.gru(x)        # h_n: (1, B, E)
        h = h_n.squeeze(0)          # (B, E)

        z = self.to_complex(h)      # (B, 2D)
        real, imag = torch.chunk(z, 2, dim=-1)  # (B, D), (B, D)
        psi0 = torch.complex(real, imag)        # (B, D)
        psi0 = complex_normalize(psi0)
        return psi0


# -----------------------------
# MZI mesh layer (optics-compatible unitary)
# -----------------------------
    """
    One layer of a 1D MZI mesh acting on NUM_MODES modes.

    We arrange parameterized 2x2 unitaries across disjoint pairs of modes.
    Two passes (even pairs and odd pairs) approximate a full mesh.

    For each pair (i, j), we use parameters (theta, phi, rho) to build:

      [ e^{i phi}  0 ] [ cos(theta)   i sin(theta) ] [ e^{i rho}   0 ]
      [   0      1 ] [ i sin(theta)   cos(theta) ] [    0       1 ]

    (Up to some parameterization variants; main point is 2x2 unitary.)
    """

class MZILayer(nn.Module):
    """
    One layer of a 1D MZI mesh acting on NUM_MODES modes.
    Vectorized implementation (no Python loops over pairs).
    """
    def __init__(self, num_modes: int):
        super().__init__()
        assert num_modes % 2 == 0, "For simplicity, use even number of modes"
        self.num_modes = num_modes

        # Global per-mode phase
        self.phase = nn.Parameter(torch.zeros(num_modes))

        # Number of disjoint pairs for even and odd passes
        num_pairs_even = num_modes // 2           # (0,1), (2,3), ..., (1022,1023)
        num_pairs_odd  = (num_modes - 1) // 2     # (1,2), (3,4), ..., (1021,1022)

        # EVEN params
        self.theta_even = nn.Parameter(torch.zeros(num_pairs_even))
        self.phi_even   = nn.Parameter(torch.zeros(num_pairs_even))
        self.rho_even   = nn.Parameter(torch.zeros(num_pairs_even))

        # ODD params
        self.theta_odd  = nn.Parameter(torch.zeros(num_pairs_odd))
        self.phi_odd    = nn.Parameter(torch.zeros(num_pairs_odd))
        self.rho_odd    = nn.Parameter(torch.zeros(num_pairs_odd))

        # Small random init for the "angle" parameters
        nn.init.normal_(self.theta_even, mean=0.0, std=0.1)
        nn.init.normal_(self.theta_odd,  mean=0.0, std=0.1)

    def _apply_pair_layer(
        self,
        psi: torch.Tensor,
        theta: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        """
        Vectorized 2x2 unitaries over all disjoint pairs in one go.

        offset = 0  → (0,1), (2,3), ...
        offset = 1  → (1,2), (3,4), ...
        """
        B, D = psi.shape
        assert D == self.num_modes

        inp = psi
        out = psi.clone()

        # Pair indices as in range(offset, D-1, 2)
        idx0 = torch.arange(offset, D - 1, 2, device=psi.device)
        idx1 = idx0 + 1
        P = idx0.shape[0]

        # Sanity: parameter length must match number of pairs
        assert theta.shape[0] == P
        assert phi.shape[0] == P
        assert rho.shape[0] == P

        # (B, P) complex
        v0 = inp[:, idx0]
        v1 = inp[:, idx1]

        # (P,) real
        t = theta
        p = phi
        r = rho

        c = torch.cos(t)
        s = torch.sin(t)
        exp_ip = torch.exp(1j * p)
        exp_ir = torch.exp(1j * r)

        # (P,) complex
        u11 = exp_ip * c * exp_ir
        u12 = exp_ip * 1j * s
        u21 = 1j * s * exp_ir
        u22 = c

        # Broadcast to (B, P)
        u11b = u11.unsqueeze(0)  # (1, P)
        u12b = u12.unsqueeze(0)
        u21b = u21.unsqueeze(0)
        u22b = u22.unsqueeze(0)

        v0_new = u11b * v0 + u12b * v1
        v1_new = u21b * v0 + u22b * v1

        out[:, idx0] = v0_new
        out[:, idx1] = v1_new
        return out

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        # Per-mode phase
        exp_phase = torch.exp(1j * self.phase.to(psi.real.dtype)).to(psi.device)
        psi = psi * exp_phase  # (B, D), broadcast over batch

        # Even pairs: (0,1), (2,3), ...
        psi = self._apply_pair_layer(
            psi, self.theta_even, self.phi_even, self.rho_even, offset=0
        )
        # Odd pairs: (1,2), (3,4), ...
        psi = self._apply_pair_layer(
            psi, self.theta_odd,  self.phi_odd,  self.rho_odd,  offset=1
        )

        # Renormalize to control drift
        psi = complex_normalize(psi)
        return psi



class QuantumCore(nn.Module):
    """
    Stack of MZI layers forming a deep unitary U.
    """
    def __init__(self, num_modes: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([MZILayer(num_modes) for _ in range(num_layers)])

    def forward(self, psi0: torch.Tensor) -> torch.Tensor:
        psi = psi0
        for layer in self.layers:
            psi = layer(psi)
        return psi


# -----------------------------
# Readout layer (Born rule)
# -----------------------------

class QuantumReadout(nn.Module):
    def __init__(self, num_modes: int, vocab_size: int):
        super().__init__()
        self.R = nn.Parameter(
            torch.randn(vocab_size, num_modes, dtype=torch.cfloat) * 0.01
        )

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        # psi: (B, D) complex
        # R:   (V, D) complex
        amps = psi @ self.R.T   # (B, V)
        log_probs = born_log_probs(amps)
        return log_probs


# -----------------------------
# Full Quantum LLM
# -----------------------------

class QuantumLM(nn.Module):
    """
    Quantum-style language model:

      tokens (context) -> psi0 (complex state)
                         -> QuantumCore (unitary)
                         -> QuantumReadout (Born rule)
                         -> log_probs over vocab
    """
    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = EMBED_DIM,
                 num_modes: int = NUM_MODES,
                 num_layers: int = NUM_MZI_LAYERS):
        super().__init__()
        self.encoder = ContextEncoder(vocab_size, embed_dim, num_modes)
        self.core = QuantumCore(num_modes, num_layers)
        self.readout = QuantumReadout(num_modes, vocab_size)

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        """
        context_tokens: (B, T) int64
        returns: log_probs: (B, V)
        """
        psi0 = self.encoder(context_tokens)   # (B, D) complex
        psif = self.core(psi0)                # (B, D) complex
        log_probs = self.readout(psif)        # (B, V) real
        return log_probs
