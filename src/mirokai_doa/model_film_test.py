import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# FiLM: time-wise (per step) with near-identity init
# -----------------------------
class SRPFiLM1D(nn.Module):
    """
    FiLM conditioning over per-time embeddings:
      srp_bt: [B,T,K]  ->  MLP(K -> 2C) -> (Δγ, β) [B,T,C]
      x_btC:  [B,T,C]  ->  (1+Δγ)*x + β
    """
    def __init__(self, C, K=72, hidden=64, use_layernorm=True):
        super().__init__()
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(C)

        self.mlp = nn.Sequential(
            nn.Linear(K, hidden, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2 * C, bias=True),
        )
        # Near-identity init on last layer so Δγ, β start ~ 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, srp_bt, x_btC):
        # srp_bt: [B,T,K], x_btC: [B,T,C]
        B, T, C = x_btC.shape
        if self.use_layernorm:
            x_btC = self.ln(x_btC)

        gb = self.mlp(srp_bt.reshape(B * T, -1)).view(B, T, 2 * C)
        dgamma, beta = gb.chunk(2, dim=-1)  # [B,T,C], [B,T,C]
        return (1.0 + dgamma) * x_btC + beta


# -----------------------------
# Backbone: Conv3d over (C-depth, T, F) to pool C & F and create filters
# -----------------------------
class CFPoolConvBackbone(nn.Module):
    """
    Input:  x [B, in_ch=12, T, F]   (e.g., T=16, F=257)
    Steps:
      - unsqueeze to [B, 1, C, T, F] so 'C' becomes the depth (D) axis for Conv3d
      - Conv3d stack with stride (2,1,2) to halve C and F, keep T:
          1) 1 -> 16  : [B,16, 6, T,129]
          2) 16 -> 32 : [B,32, 3, T, 65]
          3) 32 -> 64 : [B,64, 2, T, 33]
      - Conv3d 1x1x1: 64 -> 128 channels:            [B,128,2,T,33]
      - AdaptiveAvgPool3d to set depth=1 and F'=32:  [B,128,1,T,32]
      - squeeze depth -> [B,128,T,32]
      - per-time flatten (128*32=4096) and MLP: 4096 -> 512 -> 512 -> out_dim (=C)
    Output: per-time embeddings z [B, T, out_dim]
    """
    def __init__(self, in_ch=12, out_dim=128, hidden=512):
        super().__init__()
        # 3 strided Conv3d stages (pool C & F, keep T)
        def conv_bn_silu(cin, cout, k, s, p):
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm3d(cout),
                nn.SiLU(inplace=True),
            )

        # treat original C (12) as depth for 3D conv => first conv takes 1 "channel"
        self.conv1 = conv_bn_silu(1, 16, k=(3,1,3), s=(2,1,2), p=(1,0,1))   # -> [B,16, 6,T,129]
        self.conv2 = conv_bn_silu(16, 32, k=(3,1,3), s=(2,1,2), p=(1,0,1))  # -> [B,32, 3,T, 65]
        self.conv3 = conv_bn_silu(32, 64, k=(3,1,3), s=(2,1,2), p=(1,0,1))  # -> [B,64, 2,T, 33]

        # lift channels to 128 without changing (depth=2, T, F')
        self.conv4 = conv_bn_silu(64, 128, k=(1,1,1), s=(1,1,1), p=(0,0,0)) # -> [B,128,2,T,33]

        # collapse depth to 1 and force F' = 32 (works for F=257 or 256)
        self.poolD = nn.AdaptiveAvgPool3d((1, None, 32))                    # -> [B,128,1,T,32]

        # fixed flatten size after pooling: 128 * 32
        self.flat_dim = 128 * 32

        # MLP: 4096 -> 512 -> 512 -> out_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_dim, hidden, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=True),
        )
        self.out_dim = out_dim
        self.hidden = hidden

    def forward(self, x):  # x: [B, in_ch, T, F] with in_ch=12, F can be 257
        B, C, T, Freq = x.shape
        assert C >= 1, "in_ch must be >=1"

        # Arrange to [B,1,Depth=C,T,Freq] for Conv3d over (Depth=C, T, F)
        x5 = x.unsqueeze(1)  # [B,1,C,T,F]
        x5 = self.conv1(x5)
        x5 = self.conv2(x5)
        x5 = self.conv3(x5)
        x5 = self.conv4(x5)             # [B,128,2,T,33]
        x5 = self.poolD(x5)             # [B,128,1,T,32]
        B, Ch, Dp, T, Fp = x5.shape     # Dp == 1

        x4 = x5.squeeze(2)              # [B,128,T,32]

        # flatten (channels * F') per time
        x_flat = x4.permute(0, 2, 1, 3).contiguous().view(B, T, Ch * Fp)  # [B,T,128*32=4096]

        z = self.mlp(x_flat)            # [B,T,out_dim]
        return z


# -----------------------------
# Mixer block over [B,C,T]
# Depthwise Conv in time + channel MLP (pre-norm, residual)
# -----------------------------
class Mixer1DBlock(nn.Module):
    """
    Token mixing (depthwise Conv1d over time) + Channel MLP.
    Input/Output: h [B, C, T]
    """
    def __init__(self, C, expansion=4, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2

        # Token mixing
        self.ln_t = nn.LayerNorm(C)
        self.dwconv = nn.Conv1d(C, C, kernel_size=kernel_size, padding=pad,
                                dilation=dilation, groups=C, bias=False)
        self.pwconv = nn.Conv1d(C, C, kernel_size=1, bias=True)
        self.drop1 = nn.Dropout(dropout)

        # Channel mixing
        self.ln_c = nn.LayerNorm(C)
        self.mlp_c = nn.Sequential(
            nn.Linear(C, expansion * C, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(expansion * C, C, bias=True),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, h):  # [B,C,T]
        B, C, T = h.shape

        # Token mixing (pre-norm in [B,T,C] space)
        y = h.transpose(1, 2)           # [B,T,C]
        y = self.ln_t(y)                # LN over C
        y = y.transpose(1, 2)           # [B,C,T]
        y = self.dwconv(y)              # depthwise conv over T
        y = self.pwconv(y)              # pointwise restore mixing
        y = self.drop1(y)
        h = h + y

        # Channel mixing (pre-norm)
        y2 = h.transpose(1, 2)          # [B,T,C]
        y2 = self.ln_c(y2)
        y2 = self.mlp_c(y2)             # per-time channel MLP
        y2 = y2.transpose(1, 2)         # [B,C,T]
        y2 = self.drop2(y2)
        h = h + y2
        return h


# -----------------------------
# Full model: Backbone -> FiLM -> Mixer -> Head
# -----------------------------
class FiLMMixerSRP(nn.Module):
    """
    x:   [B, in_ch=12, T, F]      (e.g., T=16, F=257)
    srp: [B, T, K]
    out: logits [B, T, K]  (+ optional vMF head)
    """
    def __init__(self, in_ch=12, K=72, C=128, nblk=4, vmf_head=False,
                 mixer_expansion=4, mixer_drop=0.0):
        super().__init__()
        self.C = C
        self.K = K
        self.vmf_head = vmf_head

        # Backbone to produce per-time embeddings [B,T,C]
        self.backbone = CFPoolConvBackbone(in_ch=in_ch, out_dim=C, hidden=512)

        # FiLM from SRP [B,T,K] -> (Δγ, β) in [B,T,C]
        self.film = SRPFiLM1D(C=C, K=K, hidden=64, use_layernorm=True)

        # Temporal mixer over [B,C,T]
        self.blocks = nn.ModuleList([
            Mixer1DBlock(C=C, expansion=mixer_expansion,
                         kernel_size=3, dilation=(2 ** i), dropout=mixer_drop)
            for i in range(nblk)
        ])

        # Head to classes
        self.head = nn.Linear(C, K)

        if self.vmf_head:
            self.vmf = nn.Linear(C, 3)

    def forward(self, x, srp):
        """
        x:   [B,in_ch,T,F]
        srp: [B,T,K]
        returns:
            logits [B,T,K]  (and optionally (mu, kappa))
        """
        # 1) Backbone → per-time embeddings
        z = self.backbone(x)                 # [B,T,C]
        assert z.size(-1) == self.C, f"Backbone out_dim {z.size(-1)} != C {self.C}"

        # 2) FiLM from SRP (ensure same device)
        srp = srp.to(z.device)
        z = self.film(srp, z)                # [B,T,C]

        # 3) Temporal mixing on [B,C,T]
        h = z.transpose(1, 2).contiguous()   # [B,C,T]
        for blk in self.blocks:
            h = blk(h)                       # residuals inside the block

        # 4) Back to [B,T,C] and classify per time
        z = h.transpose(1, 2).contiguous()   # [B,T,C]
        logits = self.head(z)                # [B,T,K]

        if not self.vmf_head:
            return logits

        vm = self.vmf(z)                     # [B,T,3]
        mu = F.normalize(vm[..., :2], dim=-1)
        kappa = F.softplus(vm[..., 2:]) + 1e-4
        return logits, (mu, kappa)


# -----------------------------
# (Optional) quick shape check
# -----------------------------
if __name__ == "__main__":
    B, C_in, T, Freq, K = 2, 12, 16, 257, 72
    x = torch.randn(B, C_in, T, Freq)
    srp = torch.randn(B, T, K)
    model = FiLMMixerSRP(in_ch=C_in, K=K, C=128, nblk=4, vmf_head=False)
    out = model(x, srp)
    print("logits:", out.shape)  # expected [B, T, K]
