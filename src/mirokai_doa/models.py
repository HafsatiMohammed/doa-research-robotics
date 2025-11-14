# Optional: keep this import in case your project needs it, but don't fail if it's absent.
try:
    from .utils_model import *  # noqa
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# DoA-style Backbone
# =========================

class ConvBlock(nn.Module):
    """
    Same ConvBlock as in doa_model.py:
      Conv2d(in_ch -> out_ch, 3x3, padding=1) + BN + ReLU + MaxPool2d(pool_kernel).
    """
    def __init__(self, in_ch, out_ch, pool_kernel):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel)
        )

    def forward(self, x):
        return self.net(x)


class DoABackbone(nn.Module):
    """
    DoA-style backbone (copied in spirit from doa_model.py), but input is [B,T,F,C]:

    Input:  x [B, T, F, C_in]
        -> permute to [B, C_in, T, F]
        -> ConvBlock(C_in -> 64, pool (1,8))
        -> ConvBlock(64 -> 64, pool (1,8))
        -> ConvBlock(64 -> 64, pool (1,4))
        -> x [B, 64, T, F']   (T unchanged, F' pooled)
        -> reshape to [B, T, 64 * F']
        -> BiLSTM  -> [B, T, out_dim]

    Output: per-time embeddings [B, T, out_dim]
    """
    def __init__(
        self,
        in_ch: int = 12,
        conv_channels: int = 64,
        out_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.c1 = ConvBlock(in_ch,         conv_channels, pool_kernel=(1, 8))
        self.c2 = ConvBlock(conv_channels, conv_channels, pool_kernel=(1, 8))
        self.c3 = ConvBlock(conv_channels, conv_channels, pool_kernel=(1, 4))

        self.rnn = None  # lazily built on first forward when F' is known
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

    def _build_rnn(self, feat_per_timestep: int, device, dtype):
        dirs = 2 if self.bidirectional else 1
        if self.out_dim % dirs != 0:
            raise ValueError(
                f"out_dim={self.out_dim} must be divisible by num_directions={dirs} "
                f"(for BiLSTM with {dirs} directions)."
            )
        hidden = self.out_dim // dirs

        self.rnn = nn.LSTM(
            input_size=feat_per_timestep,
            hidden_size=hidden,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        ).to(device=device, dtype=dtype)

    def forward(self, x):  # x: [B, T, F, C_in]
        B, T, F, C_in = x.shape

        # -> [B, C_in, T, F] for Conv2d
        x = x.permute(0, 3, 1, 2).contiguous()

        # Conv + pooling stack (time dimension T is preserved by pool_kernel=(1, k))
        x = self.c1(x)  # [B, 64, T, F1]
        x = self.c2(x)  # [B, 64, T, F2]
        x = self.c3(x)  # [B, 64, T, F3]

        B2, C, T2, W = x.shape
        assert B2 == B and T2 == T, "Time dimension T must be preserved by pooling."

        # Flatten feature map per time step: (C * W)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * W)  # [B, T, C*W]
        feat_per_timestep = x.size(-1)

        # Lazy build RNN with the correct input size (depends on F)
        if self.rnn is None:
            self._build_rnn(feat_per_timestep, x.device, x.dtype)

        x, _ = self.rnn(x)  # [B, T, out_dim]
        return x


# =========================
# Shared Building Blocks
# =========================

class DWConv1d(nn.Module):
    """Depthwise 1D conv (expects [B,C,L])."""
    def __init__(self, c, k=3, dilation=1):
        super().__init__()
        self.dw = nn.Conv1d(c, c, k, padding=dilation * (k // 2),
                            dilation=dilation, groups=c, bias=False)
        self.pw = nn.Conv1d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm1d(c)
        self.act = nn.SiLU(True)

    def forward(self, x):  # [B,C,L]
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class AdditiveCrossAttn(nn.Module):
    """TRT-friendly additive cross-attention (Fastformer-style). q from SRP tokens, k/v from feature tokens."""
    def __init__(self, d, heads=2):
        super().__init__()
        assert d % heads == 0, "d must be divisible by heads"
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.heads = heads

    def forward(self, q_tok, kv_tok):
        """
        q_tok: [B,M,D]; kv_tok: [B,N,D]
        Returns: [B,M,D]
        """
        B, M, D = q_tok.shape
        N = kv_tok.shape[1]
        H = self.heads
        Dh = D // H

        Q = self.q(q_tok).view(B, M, H, Dh)   # [B,M,H,Dh]
        K = self.k(kv_tok).view(B, N, H, Dh)  # [B,N,H,Dh]
        V = self.v(kv_tok).view(B, N, H, Dh)  # [B,N,H,Dh]

        # Global key summary per head
        k_ctx = K.mean(dim=1, keepdim=True)                # [B,1,H,Dh]
        # Gate queries
        q_gate = torch.sigmoid(Q)                          # [B,M,H,Dh]
        # Reweight values by global key and query gates
        v_ctx = (V * k_ctx).mean(dim=1, keepdim=True)      # [B,1,H,Dh]
        out = (q_gate * v_ctx).reshape(B, M, D)            # [B,M,D]
        return self.proj(out)                              # [B,M,D]


class SRPPrototypes(nn.Module):
    """Per-time SRP projection into M prototypes with embedding dim d."""
    def __init__(self, K=72, M=12, d=128):
        super().__init__()
        self.lin1 = nn.Linear(K, 2 * M)
        self.act = nn.SiLU(True)
        self.lin2 = nn.Linear(2 * M, M * d)
        self.M, self.d = M, d

    def forward(self, srp_bt):  # [B,T,K]
        B, T, K = srp_bt.shape
        x = self.lin2(self.act(self.lin1(srp_bt)))         # [B,T,M*d]
        x = x.view(B, T, self.M, self.d)                   # [B,T,M,D]
        return x


class SRPFiLM1D(nn.Module):
    """FiLM conditioning from SRP over per-time embeddings: x [B,T,C] -> FiLM(x)."""
    def __init__(self, C, K=72):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(K, 64),
            nn.SiLU(True),
            nn.Linear(64, 2 * C)
        )

    def forward(self, srp_bt, x_btC):
        # srp_bt: [B,T,K], x_btC: [B,T,C]
        B, T, C = x_btC.shape
        g = self.mlp(srp_bt.view(B * T, -1)).view(B, T, 2 * C)  # [B,T,2C]
        gamma, beta = g.chunk(2, dim=-1)                        # [B,T,C], [B,T,C]
        return gamma * x_btC + beta


class Mixer1DBlock(nn.Module):
    """Time-only Mixer block operating on [B,C,T]."""
    def __init__(self, C, T_dil=1):
        super().__init__()
        self.tmix = DWConv1d(C, k=3, dilation=T_dil)  # [B,C,T] -> [B,C,T]
        self.pw = nn.Conv1d(C, C, 1, bias=False)
        self.bn = nn.BatchNorm1d(C)
        self.act = nn.SiLU(True)

    def forward(self, x):  # x: [B,C,T]
        y = self.tmix(x)
        y = self.bn(self.pw(self.act(y + x)))  # residual + PW
        return y


# =========================
# Models with DoA-style Backbone
# =========================

class SCATTiny(nn.Module):
    """
    SRP-conditioned additive cross-transformer with DoA-style conv+BiLSTM backbone.

    Input:
        x:   [B, T, F, C]  (batch, time, freq, channels)
        srp: [B, T, K]

    Output:
        logits: [B, T, K]  (and optional vMF params if vmf_head=True)
    """
    def __init__(self, in_ch=12, K=72, C=128, M=12, heads=2, vmf_head=False):
        super().__init__()
        self.K, self.M, self.C = K, M, C
        self.vmf_head = vmf_head

        self.backbone = DoABackbone(in_ch=in_ch, out_dim=C)
        self.srp_proto = SRPPrototypes(K=K, M=M, d=C)
        self.cross = AdditiveCrossAttn(d=C, heads=heads)
        self.cls = nn.Linear(C, K)
        if self.vmf_head:
            self.vmf = nn.Linear(C, 3)

    def forward(self, x, srp):
        """
        x:   [B,T,F,C]
        srp: [B,T,K]
        """
        B, T, F, C_in = x.shape

        # Backbone per-time embeddings (backbone handles permute)
        feats_btC = self.backbone(x)             # [B,T,C]
        B, T, C = feats_btC.shape

        # kv tokens per time: single token
        kv = feats_btC.unsqueeze(2)              # [B,T,1,C]
        kv = kv.reshape(B * T, 1, C)             # [B*T, 1, C]

        # SRP queries: [B,T,M,C] -> [B*T,M,C]
        q = self.srp_proto(srp).reshape(B * T, self.M, C)

        # Cross-attend (additive): [B*T,M,C]
        z = self.cross(q, kv)                    # [B*T,M,C]

        # Aggregate M prototypes -> one vector per time
        z = z.mean(dim=1)                        # [B*T,C]

        # Class logits per time
        logits = self.cls(z).view(B, T, self.K)  # [B,T,K]

        if not self.vmf_head:
            return logits

        vm = self.vmf(z).view(B, T, 3)
        mu = F.normalize(vm[..., :2], dim=-1)
        kappa = F.softplus(vm[..., 2:]) + 1e-4
        return logits, (mu, kappa)


class FiLMMixerSRP(nn.Module):
    """
    Time-only Mixer with SRP FiLM conditioning and DoA-style backbone.

    Input:
        x:   [B,T,F,C]
        srp: [B,T,K]

    Output:
        logits: [B,T,K] (and optional vMF params)
    """
    def __init__(self, in_ch=12, K=72, C=128, nblk=4, vmf_head=False):
        super().__init__()
        self.K = K
        self.C = C
        self.vmf_head = vmf_head

        # Backbone to get [B,T,C]
        self.backbone = DoABackbone(in_ch=in_ch, out_dim=C)

        self.film = SRPFiLM1D(C, K=K)
        self.blocks = nn.ModuleList([Mixer1DBlock(C, T_dil=2 ** i) for i in range(nblk)])
        self.head = nn.Linear(C, K)
        if self.vmf_head:
            self.vmf = nn.Linear(C, 3)

    def forward(self, x, srp):
        """
        x:   [B,T,F,C]
        srp: [B,T,K]
        """
        B, T, F, C_in = x.shape

        # Backbone → per-time embeddings
        z = self.backbone(x)                     # [B,T,C]

        # FiLM from SRP
        z = self.film(srp, z)                    # [B,T,C]

        # Temporal mixing on [B,C,T]
        h = z.transpose(1, 2).contiguous()       # [B,C,T]
        for b in self.blocks:
            h = b(h) + h                         # residual

        # Back to [B,T,C]
        z = h.transpose(1, 2).contiguous()       # [B,T,C]

        # Per-time classification
        logits = self.head(z)                    # [B,T,K]

        if not self.vmf_head:
            return logits

        vm = self.vmf(z)                         # [B,T,3]
        mu = F.normalize(vm[..., :2], dim=-1)
        kappa = F.softplus(vm[..., 2:]) + 1e-4
        return logits, (mu, kappa)


class RetentiveCell(nn.Module):
    """
    h_t = alpha_t ⊙ h_{t-1} + (1 - alpha_t) ⊙ f(x_t)
    alpha_t comes from SRP reliability (peak & curvature)
    """
    def __init__(self, C):
        super().__init__()
        self.f = nn.Linear(C, C, bias=False)
        self.g = nn.Linear(2, C)  # inputs: [peak, curvature] -> alpha per channel
        self.act = nn.SiLU(True)

    def forward(self, x_t, srp_t, h_prev):
        # x_t: [B,C]; srp_t: [B,K]; h_prev: [B,C]
        with torch.no_grad():
            q = (srp_t / (srp_t.sum(dim=-1, keepdim=True) + 1e-8))
            peak, _ = q.max(dim=-1, keepdim=True)                          # [B,1]
            q_p = torch.roll(q, 1, dims=-1)
            q_n = torch.roll(q, -1, dims=-1)
            curv = (q_p - 2 * q + q_n).abs().mean(dim=-1, keepdim=True)    # [B,1]
            rel = torch.cat([peak, curv], dim=-1)                           # [B,2]

        alpha = torch.sigmoid(self.g(rel))                                  # [B,C] in (0,1)
        xproj = self.act(self.f(x_t))                                       # [B,C]
        h = alpha * h_prev + (1 - alpha) * xproj
        return h


class ReTiNDoA(nn.Module):
    """
    Retentive cell unrolled over time with DoA-style backbone.
    - Backbone: DoABackbone -> per-time embeddings [B,T,C].

    Input:
        x:   [B,T,F,C]
        srp: [B,T,K]

    Output:
        logits: [B,T,K], delta: [B,T,1], hT: [B,C]
    """
    def __init__(self, in_ch=12, K=72, C=96):
        super().__init__()
        self.K = K
        self.C = C
        self.backbone = DoABackbone(in_ch=in_ch, out_dim=C)
        self.cell = RetentiveCell(C)
        self.head = nn.Linear(C, K)
        self.off = nn.Linear(C, 1)  # optional Δθ regressor (radians ~[-0.13, 0.13])

    def forward(self, x, srp, h0=None):
        """
        x:   [B,T,F,C]
        srp: [B,T,K]
        returns logits [B,T,K], delta [B,T,1], hT [B,C]
        """
        B, T, F, C_in = x.shape

        # Backbone → per-time embeddings
        z = self.backbone(x)                     # [B,T,C]

        if h0 is None:
            h_state = torch.zeros(B, self.C, device=x.device, dtype=x.dtype)
        else:
            h_state = h0

        logits_list = []
        delta_list = []
        for t in range(T):                       # unrolled over time
            h_state = self.cell(z[:, t, :], srp[:, t, :], h_state)  # [B,C]
            logits_list.append(self.head(h_state))                  # [B,K]
            delta_list.append(self.off(h_state))                    # [B,1]

        logits = torch.stack(logits_list, dim=1)                    # [B,T,K]
        delta = torch.stack(delta_list, dim=1)                      # [B,T,1]
        return logits, delta, h_state


# =========================
# Smoke Test
# =========================

if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, Freq, C_in = 64, 25, 513, 12
    K = 72

    # Your real layout: [B,T,F,C]
    x = torch.randn(B, T, Freq, C_in).to('cuda')
    srp = torch.randn(B, T, K).to('cuda')

    print("Input:", x.shape)  # [B,T,F,C]

    # Model 1
    model_1 = SCATTiny(in_ch=C_in, K=K, C=128, M=12, heads=2, vmf_head=False).to('cuda')
    logits_1 = model_1(x, srp)
    print("SCATTiny logits:", logits_1.shape)  # [B,T,K]

    # Model 2
    model_2 = FiLMMixerSRP(in_ch=C_in, K=K, C=128, nblk=4, vmf_head=False).to('cuda')
    logits_2 = model_2(x, srp)
    print("FiLMMixerSRP logits:", logits_2.shape)  # [B,T,K]

    # Model 3
    model_3 = ReTiNDoA(in_ch=C_in, K=K, C=96).to('cuda')
    logits_3, delta, h_state = model_3(x, srp)
    print("ReTiNDoA logits:", logits_3.shape, " delta:", delta.shape, " hT:", h_state.shape)

    # Also verify backbone shapes directly
    bb = DoABackbone(in_ch=C_in, out_dim=128).to('cuda')
    z = bb(x)  # [B,T,128]
    print("Backbone output:", z.shape)
