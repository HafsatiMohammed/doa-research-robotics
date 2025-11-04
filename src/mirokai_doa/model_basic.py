import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
class TFPoolClassifierNoCond(nn.Module):
    """
    Input:
        x: [B, 12, T=16, F=257]
    Conv channels:
        32 -> 64 -> 32 -> 64 -> 128
    Pooling:
        AvgPool2d(kernel=2, stride=2) after the first four conv blocks (pools T and F)
        Shapes (T,F): 16,257 -> 8,128 -> 4,64 -> 2,32 -> 1,16
    Head (MLP):
        128 -> 256 -> 128 -> K (=72)
    Output:
        logits: [B, K]
    """
    def __init__(self, in_ch=12, K=72, dropout=0.0):
        super().__init__()

        def conv_bn_silu(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.SiLU(inplace=True),
            )

        chs = [32, 64, 32, 64, 128]
        self.block1 = conv_bn_silu(in_ch,    chs[0])  # [B, 32, 16, 257]
        self.block2 = conv_bn_silu(chs[0],   chs[1])  # [B, 64,  8, 128] after pool
        self.block3 = conv_bn_silu(chs[1],   chs[2])  # [B, 32,  4,  64] after pool
        self.block4 = conv_bn_silu(chs[2],   chs[3])  # [B, 64,  2,  32] after pool
        self.block5 = conv_bn_silu(chs[3],   chs[4])  # [B,128, 1,  16] after pool

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)   # pools T and F
        self.gap  = nn.AdaptiveAvgPool2d((1, 1))            # -> [B,128,1,1]

        # 128 -> 256 -> 128 -> K
        self.mlp = nn.Sequential(
            nn.Linear(chs[-1], 256, bias=True),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(128, K, bias=True),
        )



        p = 1.75 / 72.0     # tweak if your max-3 is often <3
        b =  math.log(p / (1.0 - p))
        nn.init.constant_(self.mlp[-1].bias, b)

    def forward(self, x, srp):                 # x: [B,12,16,257]
        x = self.block1(x)               # [B,32,16,257]
        x = self.pool(x)                 # [B,32, 8,128]

        x = self.block2(x)               # [B,64, 8,128]
        x = self.pool(x)                 # [B,64, 4, 64]

        x = self.block3(x)               # [B,32, 4, 64]
        x = self.pool(x)                 # [B,32, 2, 32]

        x = self.block4(x)               # [B,64, 2, 32]
        x = self.pool(x)                 # [B,64, 1, 16]

        x = self.block5(x)               # [B,128,1,16]
        x = self.gap(x)                  # [B,128,1,1]

        x = x.flatten(1)                 # [B,128]
        logits = self.mlp(x)             # [B,72]
        return torch.sigmoid(logits) #logits


# Quick shape check
if __name__ == "__main__":
    B = 4
    x = torch.randn(B, 12, 16, 257)
    model = TFPoolClassifierNoCond(K=72)
    y = model(x)
    print(y.shape)  # torch.Size([4, 72])
