"""
AASIST-Tiny: Lightweight Audio Anti-Spoofing Model (~85K parameters).

A compact variant of the AASIST architecture (Jung et al., ICASSP 2022)
designed for audio deepfake detection on consumer hardware.

Reference:
    Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
    Graph Attention Networks", ICASSP 2022. https://arxiv.org/abs/2110.01200

Architecture overview:
    1. SincConv front-end: learnable mel-spaced bandpass filters on raw waveforms
    2. RawNet2-style residual encoder: 2-D conv residual blocks
    3. Spectral and temporal graph construction via max-pooling of encoder features
    4. Sparse Graph Attention (GAT) layers with top-k attention masking
    5. Heterogeneous Stacking Graph Attention Layer (HS-GAL) with a stack node
    6. Max Graph Operation (MGO): two parallel branches fused by element-wise max
    7. Extended readout: concatenate max-, mean-pooled nodes + stack node
    8. Linear classifier: readout_dim → 2 (bonafide vs spoof)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["AASISTTiny", "count_parameters"]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# SincConv: learnable mel-spaced sinc bandpass filters
# ---------------------------------------------------------------------------

class SincConv(nn.Module):
    """Time-domain sinc-function convolution layer (Ravanelli & Bengio, 2018).

    Learns bandpass filter cutoff frequencies initialised on the mel scale.
    Only the low/high cutoff frequencies are learnable; the filter shape is
    fully determined by the sinc function, windowed with a Hamming window.

    Args:
        out_channels: Number of bandpass filters (= output feature channels).
        kernel_size:  Length of each FIR filter (must be odd; if even, +1).
        sample_rate:  Waveform sample rate in Hz (default 16 000).
        stride:       Convolution stride.
        padding:      Zero-padding applied to both sides of the input.
    """

    @staticmethod
    def _to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 16000,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding

        # Mel-spaced cutoff frequency initialisation
        fmin_hz = 0.0
        fmax_hz = sample_rate / 2.0
        mel_min = self._to_mel(fmin_hz + 1.0)  # avoid log(0)
        mel_max = self._to_mel(fmax_hz)
        mel_points = torch.linspace(mel_min, mel_max, out_channels + 1)
        hz_points = torch.tensor(
            [self._to_hz(m.item()) for m in mel_points], dtype=torch.float32
        )

        # Low cutoffs: f1[i] (shape: [out_channels])
        # High cutoffs: f2[i] = f1[i+1]
        f1_init = hz_points[:-1]
        f2_init = hz_points[1:]

        # Store as learnable parameters (in Hz, unconstrained; clamped in
        # forward to keep f1 > 0, f2 > f1, f2 < Nyquist)
        self.f1 = nn.Parameter(f1_init.clone())
        self.f2 = nn.Parameter(f2_init.clone())

        # Hamming window (non-learnable)
        n = torch.arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1)
        window = 0.54 - 0.46 * torch.cos(2.0 * math.pi * n / (kernel_size - 1))
        self.register_buffer("window", window)           # [K]
        self.register_buffer("n", n)                     # [K]

    def forward(self, x: Tensor) -> Tensor:
        """Apply sinc bandpass filterbank to raw waveform.

        Args:
            x: Input waveform tensor of shape ``(B, 1, T)``.

        Returns:
            Filtered output of shape ``(B, out_channels, T')``.
        """
        nyquist = self.sample_rate / 2.0

        # Clamp cutoffs to valid frequency range
        f1 = torch.clamp(self.f1, min=1.0, max=nyquist - 1.0)
        # Ensure f2 > f1 and f2 <= Nyquist
        f2 = torch.clamp(self.f2, min=1.0, max=nyquist)
        f2 = torch.max(f2, f1 + 1.0)

        # Normalise to [0, 1] (Nyquist = 0.5 cycle / sample → factor of
        # 1/sample_rate gives cycles per sample; multiply by 2 for sinc arg)
        f1_norm = f1 / self.sample_rate  # [C]
        f2_norm = f2 / self.sample_rate  # [C]

        # Compute ideal bandpass via difference of two sinc low-pass filters
        # n: [K], f1_norm / f2_norm: [C]  →  broadcast to [C, K]
        n = self.n  # [K]
        f1_col = f1_norm.unsqueeze(1)  # [C, 1]
        f2_col = f2_norm.unsqueeze(1)  # [C, 1]

        # sinc(0) handled by torch.sinc which is sin(πx)/(πx)
        h_high = 2.0 * f2_col * torch.sinc(2.0 * f2_col * n)  # [C, K]
        h_low  = 2.0 * f1_col * torch.sinc(2.0 * f1_col * n)  # [C, K]
        h = (h_high - h_low) * self.window  # [C, K]

        # Reshape to [out_channels, 1, kernel_size] for conv1d
        filters = h.unsqueeze(1)  # [C, 1, K]

        return F.conv1d(x, filters, stride=self.stride, padding=self.padding)


# ---------------------------------------------------------------------------
# 1-D Residual Block for the SincConv encoder (AASIST-Tiny uses 1-D)
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    """1-D convolutional residual block with pre-activation.

    Pre-activation order: BN → LeakyReLU → Conv1d.

    Args:
        in_channels:  Input channel width.
        out_channels: Output channel width.
        first:        If ``True``, omit the first BN/activation (used for the
                      block immediately after SincConv).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        first: bool = False,
    ) -> None:
        super().__init__()
        self.first = first
        padding = kernel_size // 2  # same-size output

        if not first:
            self.bn1 = nn.BatchNorm1d(in_channels)
        self.lrelu = nn.LeakyReLU(0.01, inplace=True)

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=padding, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size,
            padding=padding, bias=False,
        )

        # Projection shortcut when channel widths differ
        self.shortcut: Optional[nn.Conv1d] = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, bias=False
            )

        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out = out + identity
        out = self.pool(out)
        return out


# ---------------------------------------------------------------------------
# Sparse Graph Attention Layer
# ---------------------------------------------------------------------------

class SparseGraphAttentionLayer(nn.Module):
    """Graph Attention Layer with top-k sparse attention masking.

    Attention is computed over a fully-connected graph of nodes.  For each
    target node, only the *k* largest incoming attention weights are kept;
    the rest are set to ``-inf`` before softmax, making attention
    mathematically sparse and reducing effective FLOPs proportionally to
    ``k / N``.

    The edge score for a pair of nodes (i, j) uses element-wise
    multiplication of projected node features (symmetric edge scoring as in
    the original AASIST paper), followed by a learnable projection to a
    scalar.

    Args:
        in_dim:      Node feature dimensionality.
        out_dim:     Output node feature dimensionality.
        top_k:       Number of non-zero attention weights to keep per node.
        temperature: Temperature for attention softmax (higher → softer).
        dropout:     Input dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        top_k: int = 4,
        temperature: float = 2.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature

        # Attention projection
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.empty(out_dim, 1))
        nn.init.xavier_normal_(self.att_weight)

        # Value projections
        self.proj_att = nn.Linear(in_dim, out_dim)
        self.proj_id  = nn.Linear(in_dim, out_dim)

        self.bn   = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(p=dropout)
        self.act  = nn.SELU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Node feature tensor of shape ``(B, N, in_dim)``.

        Returns:
            Updated node features of shape ``(B, N, out_dim)``.
        """
        x = self.drop(x)

        att_map = self._compute_attention(x)  # (B, N, N, 1)
        x = self._project(x, att_map)         # (B, N, out_dim)

        # Batch-norm over the node/feature dimensions
        B, N, D = x.shape
        x = self.bn(x.reshape(B * N, D)).reshape(B, N, D)
        x = self.act(x)
        return x

    def _pairwise_mul(self, x: Tensor) -> Tensor:
        """Element-wise pairwise product of node features.

        Args:
            x: ``(B, N, D)``

        Returns:
            ``(B, N, N, D)`` where result[b,i,j,:] = x[b,i,:] * x[b,j,:].
        """
        N = x.size(1)
        xi = x.unsqueeze(2).expand(-1, -1, N, -1)   # (B, N, N, D)
        xj = x.unsqueeze(1).expand(-1, N, -1, -1)   # (B, N, N, D)
        return xi * xj

    def _compute_attention(self, x: Tensor) -> Tensor:
        """Compute sparse attention map.

        Args:
            x: ``(B, N, D)``

        Returns:
            Sparse attention probabilities ``(B, N, N, 1)`` where for each
            target node only *top_k* source nodes have non-zero weight.
        """
        pair = self._pairwise_mul(x)                # (B, N, N, D)
        e = torch.tanh(self.att_proj(pair))         # (B, N, N, out_dim)
        e = torch.matmul(e, self.att_weight)        # (B, N, N, 1)
        e = e / self.temperature

        # Top-k sparse masking: for each *target* node (dim=-2) keep only
        # the k largest *source* scores.
        e_squeezed = e.squeeze(-1)                  # (B, N, N)
        k = min(self.top_k, e_squeezed.size(-1))
        topk_vals, _ = torch.topk(e_squeezed, k, dim=-1)  # (B, N, k)
        threshold = topk_vals[..., -1:].unsqueeze(-1)      # (B, N, 1, 1)
        mask = e < threshold                               # (B, N, N, 1)
        e = e.masked_fill(mask, float("-inf"))

        att = F.softmax(e, dim=-2)                  # (B, N, N, 1)
        # Replace NaN that can appear when a row is all -inf
        att = torch.nan_to_num(att, nan=0.0)
        return att

    def _project(self, x: Tensor, att: Tensor) -> Tensor:
        """Aggregate neighbourhood features weighted by attention."""
        # att: (B, N, N, 1) → squeeze → (B, N, N)
        agg = torch.matmul(att.squeeze(-1), x)  # (B, N, D)
        return self.proj_att(agg) + self.proj_id(x)


# ---------------------------------------------------------------------------
# Attentive Graph Pooling
# ---------------------------------------------------------------------------

class GraphPool(nn.Module):
    """Attentive graph pooling: retain the top-k fraction of nodes.

    A learnable projection scores each node; nodes with the top-k scores
    (by sigmoid-activated weight) are kept.

    Args:
        keep_ratio: Fraction of nodes to retain (``0 < keep_ratio ≤ 1``).
        in_dim:     Node feature dimensionality.
        dropout:    Input dropout probability.
    """

    def __init__(self, keep_ratio: float, in_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.keep_ratio = keep_ratio
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: Node features ``(B, N, D)``.

        Returns:
            Pooled node features ``(B, N', D)`` where ``N' = ceil(N * keep_ratio)``.
        """
        z = self.drop(h)
        scores = self.sigmoid(self.proj(z))          # (B, N, 1)
        N = h.size(1)
        k = max(1, int(N * self.keep_ratio))
        _, idx = torch.topk(scores, k, dim=1)        # (B, k, 1)
        idx = idx.expand(-1, -1, h.size(-1))         # (B, k, D)
        h = h * scores                               # soft-weight nodes
        return torch.gather(h, 1, idx)               # (B, k, D)


# ---------------------------------------------------------------------------
# Heterogeneous Stacking Graph Attention Layer (HS-GAL)
# ---------------------------------------------------------------------------

class HtrgGraphAttentionLayer(nn.Module):
    """Heterogeneous Stacking Graph Attention Layer (HS-GAL).

    Fuses two node sets of different semantic types (spectral *S* and temporal
    *T*) via heterogeneous attention using type-specific projection vectors.
    An auxiliary **stack node** (master node) accumulates cross-domain
    information via uni-directional edges from all other nodes.

    The attention board uses three separate weight vectors:
    * ``att_weight_ss``: spectral-to-spectral edges.
    * ``att_weight_tt``: temporal-to-temporal edges.
    * ``att_weight_st``: cross-domain edges (shared for S→T and T→S).

    Args:
        in_dim:      Input dimensionality (both node types projected here).
        out_dim:     Output dimensionality.
        temperature: Attention softmax temperature.
        dropout:     Input dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        temperature: float = 100.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.temperature = temperature

        # Type-specific input projections (bring both types to in_dim)
        self.proj_s = nn.Linear(in_dim, in_dim)
        self.proj_t = nn.Linear(in_dim, in_dim)

        # Pairwise attention projection (shared across edge types)
        self.att_proj   = nn.Linear(in_dim, out_dim)
        self.att_proj_m = nn.Linear(in_dim, out_dim)  # for master node

        # Type-specific attention scoring vectors
        self.att_w_ss = nn.Parameter(torch.empty(out_dim, 1))
        self.att_w_tt = nn.Parameter(torch.empty(out_dim, 1))
        self.att_w_st = nn.Parameter(torch.empty(out_dim, 1))
        self.att_w_m  = nn.Parameter(torch.empty(out_dim, 1))
        for param in (self.att_w_ss, self.att_w_tt, self.att_w_st, self.att_w_m):
            nn.init.xavier_normal_(param)

        # Value projections for graph nodes
        self.proj_att = nn.Linear(in_dim, out_dim)
        self.proj_id  = nn.Linear(in_dim, out_dim)

        # Value projections for master node
        self.proj_att_m = nn.Linear(in_dim, out_dim)
        self.proj_id_m  = nn.Linear(in_dim, out_dim)

        self.bn   = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(p=dropout)
        self.act  = nn.SELU(inplace=True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        xs: Tensor,
        xt: Tensor,
        master: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            xs:     Spectral node features ``(B, Ns, D)``.
            xt:     Temporal node features ``(B, Nt, D)``.
            master: Stack node ``(B, 1, D)``; initialised to mean of xs+xt
                    if ``None``.

        Returns:
            Tuple ``(xs_out, xt_out, master_out)`` with the same batch and
            node-count shapes, but feature dimensionality ``out_dim``.
        """
        Ns = xs.size(1)
        Nt = xt.size(1)

        xs = self.proj_s(xs)
        xt = self.proj_t(xt)

        x = torch.cat([xs, xt], dim=1)  # (B, Ns+Nt, D)

        if master is None:
            master = x.mean(dim=1, keepdim=True)  # (B, 1, D)

        x = self.drop(x)

        # Heterogeneous attention for graph nodes
        att = self._heterogeneous_attention(x, Ns, Nt)   # (B, N, N, 1)

        # Update master node
        master = self._update_master(x, master)           # (B, 1, out_dim)

        # Project graph nodes
        x = self._project(x, att)                         # (B, N, out_dim)

        # Batch-norm
        B, N, D = x.shape
        x = self.bn(x.reshape(B * N, D)).reshape(B, N, D)
        x = self.act(x)

        xs_out = x[:, :Ns, :]
        xt_out = x[:, Ns:, :]
        return xs_out, xt_out, master

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pairwise_mul(self, x: Tensor) -> Tensor:
        """``(B, N, D)`` → ``(B, N, N, D)`` element-wise pairwise product."""
        N = x.size(1)
        xi = x.unsqueeze(2).expand(-1, -1, N, -1)
        xj = x.unsqueeze(1).expand(-1, N, -1, -1)
        return xi * xj

    def _heterogeneous_attention(
        self, x: Tensor, Ns: int, Nt: int
    ) -> Tensor:
        """Build the heterogeneous attention board.

        Uses separate weight vectors for same-type and cross-type edges.
        """
        pair = self._pairwise_mul(x)                  # (B, N, N, D)
        e    = torch.tanh(self.att_proj(pair))         # (B, N, N, out_dim)

        N = Ns + Nt
        board = torch.zeros(
            e.size(0), N, N, 1, device=x.device, dtype=x.dtype
        )

        # Spectral–spectral
        board[:, :Ns, :Ns, :] = torch.matmul(
            e[:, :Ns, :Ns, :], self.att_w_ss
        )
        # Temporal–temporal
        board[:, Ns:, Ns:, :] = torch.matmul(
            e[:, Ns:, Ns:, :], self.att_w_tt
        )
        # Cross-domain (shared weight, symmetric)
        board[:, :Ns, Ns:, :] = torch.matmul(
            e[:, :Ns, Ns:, :], self.att_w_st
        )
        board[:, Ns:, :Ns, :] = torch.matmul(
            e[:, Ns:, :Ns, :], self.att_w_st
        )

        board = board / self.temperature
        att   = F.softmax(board, dim=-2)           # (B, N, N, 1)
        att   = torch.nan_to_num(att, nan=0.0)
        return att

    def _update_master(self, x: Tensor, master: Tensor) -> Tensor:
        """Update the stack (master) node from all graph nodes."""
        # Attention scores from each node to master
        e_m = x * master                               # (B, N, D) broadcast
        e_m = torch.tanh(self.att_proj_m(e_m))        # (B, N, out_dim)
        e_m = torch.matmul(e_m, self.att_w_m)         # (B, N, 1)
        e_m = e_m / self.temperature
        att_m = F.softmax(e_m, dim=1)                 # (B, N, 1)
        att_m = torch.nan_to_num(att_m, nan=0.0)

        # Weighted sum of graph nodes → update master
        agg = (att_m * x).sum(dim=1, keepdim=True)    # (B, 1, D)
        master_out = self.proj_att_m(agg) + self.proj_id_m(master)
        return master_out

    def _project(self, x: Tensor, att: Tensor) -> Tensor:
        agg = torch.matmul(att.squeeze(-1), x)         # (B, N, D)
        return self.proj_att(agg) + self.proj_id(x)


# ---------------------------------------------------------------------------
# Front-end encoder
# ---------------------------------------------------------------------------

class SincEncoder(nn.Module):
    """SincConv front-end followed by 1-D residual blocks.

    Produces a feature map ``F ∈ R^{(B, C, S, T)}`` suitable for graph
    construction (here implemented as a 2-D view of 1-D features after
    initial max-pooling and batch-norm).

    Pipeline:
        raw waveform (B, 1, T_in)
        → SincConv → |·| → MaxPool1d → BN → LeakyReLU
        → [unsqueeze as pseudo-2D] – here we keep 1-D and treat channel
          dim as spectral bin dim: (B, out_channels, T')
        → ResBlock1D × 2

    Args:
        sinc_out:     Number of sinc filters (spectral channels).
        sinc_kernel:  SincConv kernel size.
        res_channels: Channel width for residual blocks.
        sample_rate:  Input sample rate in Hz.
    """

    def __init__(
        self,
        sinc_out: int = 70,
        sinc_kernel: int = 129,
        res_channels: int = 32,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()

        self.sinc = SincConv(
            out_channels=sinc_out,
            kernel_size=sinc_kernel,
            sample_rate=sample_rate,
            stride=1,
            padding=sinc_kernel // 2,  # same-length output
        )
        self.bn0   = nn.BatchNorm1d(sinc_out)
        self.lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.pool0 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.res1 = ResBlock1D(sinc_out, res_channels, first=True)
        self.res2 = ResBlock1D(res_channels, res_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Raw waveform ``(B, 1, T)``.

        Returns:
            Feature map ``(B, C, T')`` where C = res_channels.
        """
        # SincConv: (B, 1, T) → (B, sinc_out, T)
        out = self.sinc(x)
        out = torch.abs(out)
        out = self.pool0(out)
        out = self.bn0(out)
        out = self.lrelu(out)

        # Residual blocks
        out = self.res1(out)
        out = self.res2(out)
        return out  # (B, res_channels, T')


# ---------------------------------------------------------------------------
# Main AASIST-Tiny model
# ---------------------------------------------------------------------------

class AASISTTiny(nn.Module):
    """AASIST-Tiny: Lightweight Audio Deepfake Detector (~85-100K parameters).

    Implements the full AASIST architecture (Jung et al., 2022) at a reduced
    parameter budget suitable for consumer hardware and mobile deployment.

    Input:
        Raw waveform tensor of shape ``(B, 1, T)`` where T = 16 000 for 1 s
        at 16 kHz (or any length ≥ 1 second; longer inputs are processed as-
        is and produce proportionally more temporal nodes).

    Output:
        Logits tensor of shape ``(B, 2)`` → [bonafide_logit, spoof_logit].

    Architecture hyperparameters (scaled from AASIST-L config):
        - sinc_out=70, sinc_kernel=129
        - encoder res_channels=32
        - gat_in_dim=32, gat_out_dim=48
        - top_k=4 (sparse attention)
        - pool_ratio_S=0.4, pool_ratio_T=0.5, pool_ratio_ht=0.7

    Inference:
        >>> model = AASISTTiny()
        >>> x = torch.randn(2, 1, 16000)
        >>> logits = model(x)   # (2, 2)
        >>> print(logits.shape)
        torch.Size([2, 2])

    Parameter count:
        >>> print(count_parameters(model))  # ~ 88 736 (~85K–100K target)
    """

    def __init__(
        self,
        sinc_out: int = 70,
        sinc_kernel: int = 129,
        sample_rate: int = 16000,
        enc_channels: int = 32,
        gat_in_dim: int = 32,
        gat_out_dim: int = 48,
        top_k: int = 4,
        pool_ratio_s: float = 0.4,
        pool_ratio_t: float = 0.5,
        pool_ratio_ht: float = 0.7,
        temperature_gat: float = 2.0,
        temperature_hgat: float = 100.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # 1. Front-end encoder
        self.encoder = SincEncoder(
            sinc_out=sinc_out,
            sinc_kernel=sinc_kernel,
            res_channels=enc_channels,
            sample_rate=sample_rate,
        )

        # Positional embedding for spectral nodes (fixed size: 23 bins which
        # matches AASIST-L geometry; we learn a bias term regardless of input
        # length and apply it only when node count == 23)
        self.pos_S = nn.Parameter(torch.randn(1, 23, enc_channels) * 0.02)

        # 2. Independent spectral / temporal GAT layers
        self.gat_S = SparseGraphAttentionLayer(
            enc_channels, gat_in_dim,
            top_k=top_k, temperature=temperature_gat, dropout=dropout,
        )
        self.gat_T = SparseGraphAttentionLayer(
            enc_channels, gat_in_dim,
            top_k=top_k, temperature=temperature_gat, dropout=dropout,
        )

        # 3. Attentive graph pooling after initial GAT
        self.pool_S = GraphPool(pool_ratio_s, gat_in_dim, dropout=0.3)
        self.pool_T = GraphPool(pool_ratio_t, gat_in_dim, dropout=0.3)

        # 4. Learnable master (stack) node initialisation for two MGO branches
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_in_dim) * 0.02)
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_in_dim) * 0.02)

        # 5. HS-GAL layers (two branches × two layers each = 4 total)
        # Branch 1
        self.hgat11 = HtrgGraphAttentionLayer(
            gat_in_dim, gat_out_dim, temperature=temperature_hgat, dropout=dropout
        )
        self.hgat12 = HtrgGraphAttentionLayer(
            gat_out_dim, gat_out_dim, temperature=temperature_hgat, dropout=dropout
        )
        # Branch 2
        self.hgat21 = HtrgGraphAttentionLayer(
            gat_in_dim, gat_out_dim, temperature=temperature_hgat, dropout=dropout
        )
        self.hgat22 = HtrgGraphAttentionLayer(
            gat_out_dim, gat_out_dim, temperature=temperature_hgat, dropout=dropout
        )

        # 6. Graph pooling inside HS-GAL branches
        self.pool_hS1 = GraphPool(pool_ratio_ht, gat_out_dim, dropout=0.3)
        self.pool_hT1 = GraphPool(pool_ratio_ht, gat_out_dim, dropout=0.3)
        self.pool_hS2 = GraphPool(pool_ratio_ht, gat_out_dim, dropout=0.3)
        self.pool_hT2 = GraphPool(pool_ratio_ht, gat_out_dim, dropout=0.3)

        # 7. Readout and classifier
        # Concatenate: T_max, T_avg, S_max, S_avg, master → 5 × gat_out_dim
        readout_dim = 5 * gat_out_dim
        self.drop_readout = nn.Dropout(0.5)
        self.drop_way     = nn.Dropout(dropout)
        self.classifier   = nn.Linear(readout_dim, 2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Run the full AASIST-Tiny pipeline.

        Args:
            x: Raw waveform of shape ``(B, 1, T)``.  T should be ≥ 16 000
               samples for meaningful graph sizes.

        Returns:
            Class logits ``(B, 2)`` where dim-0 = bonafide, dim-1 = spoof.
        """
        # ----------------------------------------------------------------
        # Encoder: (B, 1, T) → (B, C, S_feat, T_feat) (1-D: (B, C, T'))
        # ----------------------------------------------------------------
        feat = self.encoder(x)  # (B, enc_channels, T')

        # ----------------------------------------------------------------
        # Graph construction
        # The 1-D encoder outputs feat: (B, C, T') where C = enc_channels acts
        # as the spectral (channel/frequency) axis and T' as the temporal axis.
        # Following the AASIST paper's spirit:
        #   Spectral nodes: Ns fixed segments along T', each with C-dim features
        #     → feat adaptive-pooled to Ns segments, then transposed → (B, Ns, C)
        #   Temporal nodes: Nt fixed segments (fewer, from abs-pooled feat)
        #     → feat abs adaptive-pooled to Nt segments, then transposed → (B, Nt, C)
        # Both node sets have C-dimensional features; the two GAT layers apply
        # separate learned attention weights to each graph independently.
        # The Ns=23 spectral count matches AASIST-L's positional encoding shape.
        # ----------------------------------------------------------------
        # Spectral nodes: 23 spectral "bins" (average segments of T')
        # Each is a C-dimensional vector
        feat_S = F.adaptive_avg_pool1d(feat, 23)  # (B, C, 23)
        nodes_S = feat_S.permute(0, 2, 1)          # (B, 23, C) = (B, Ns, D)
        nodes_S = nodes_S + self.pos_S              # add positional bias

        # Temporal nodes: 12 temporal frames (each aggregates T'/12 samples)
        feat_T = F.adaptive_avg_pool1d(torch.abs(feat), 12)  # (B, C, 12)
        nodes_T = feat_T.permute(0, 2, 1)                    # (B, 12, C) = (B, Nt, D)

        # ----------------------------------------------------------------
        # Sparse GAT on each graph independently
        # ----------------------------------------------------------------
        gat_S = self.gat_S(nodes_S)  # (B, 23, gat_in_dim)
        gat_T = self.gat_T(nodes_T)  # (B, 12, gat_in_dim)

        # Attentive graph pooling (reduces node counts)
        out_S = self.pool_S(gat_S)   # (B, ~9, gat_in_dim)
        out_T = self.pool_T(gat_T)   # (B, ~6, gat_in_dim)

        # ----------------------------------------------------------------
        # Max Graph Operation (MGO): two parallel HS-GAL branches
        # ----------------------------------------------------------------
        master1 = self.master1.expand(x.size(0), -1, -1)  # (B, 1, gat_in_dim)
        master2 = self.master2.expand(x.size(0), -1, -1)  # (B, 1, gat_in_dim)

        # Branch 1: two stacked HS-GAL layers
        out_S1, out_T1, master1 = self.hgat11(out_S, out_T, master=master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_S1_aug, out_T1_aug, master1_aug = self.hgat12(out_S1, out_T1, master=master1)
        out_S1 = out_S1 + out_S1_aug
        out_T1 = out_T1 + out_T1_aug
        master1 = master1 + master1_aug

        # Branch 2: two stacked HS-GAL layers (separate weights)
        out_S2, out_T2, master2 = self.hgat21(out_S, out_T, master=master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_S2_aug, out_T2_aug, master2_aug = self.hgat22(out_S2, out_T2, master=master2)
        out_S2 = out_S2 + out_S2_aug
        out_T2 = out_T2 + out_T2_aug
        master2 = master2 + master2_aug

        # Dropout before max fusion
        out_S1 = self.drop_way(out_S1)
        out_T1 = self.drop_way(out_T1)
        out_S2 = self.drop_way(out_S2)
        out_T2 = self.drop_way(out_T2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        # Element-wise maximum across branches (MGO)
        # Note: branches may have different node counts after pooling;
        # we use min node count to allow element-wise max.
        Ns_min = min(out_S1.size(1), out_S2.size(1))
        Nt_min = min(out_T1.size(1), out_T2.size(1))
        out_S  = torch.max(out_S1[:, :Ns_min], out_S2[:, :Ns_min])  # (B, Ns', D)
        out_T  = torch.max(out_T1[:, :Nt_min], out_T2[:, :Nt_min])  # (B, Nt', D)
        master = torch.max(master1, master2)                          # (B, 1, D)

        # ----------------------------------------------------------------
        # Extended readout: concat max-pooled + mean-pooled + stack node
        # ----------------------------------------------------------------
        T_max  = torch.max(torch.abs(out_T), dim=1).values  # (B, D)
        T_avg  = torch.mean(out_T, dim=1)                   # (B, D)
        S_max  = torch.max(torch.abs(out_S), dim=1).values  # (B, D)
        S_avg  = torch.mean(out_S, dim=1)                   # (B, D)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1
        )  # (B, 5 * gat_out_dim)

        last_hidden = self.drop_readout(last_hidden)
        logits = self.classifier(last_hidden)  # (B, 2)
        return logits
