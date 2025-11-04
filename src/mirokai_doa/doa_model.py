# doa_model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Model
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool_kernel):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel)  # (1,8) or (1,4)
        )
    def forward(self, x):
        return self.net(x)

class DoAEstimator(nn.Module):
    """
    Implements the paper figure:

    Input: (B, T=25, F=513, C=6)
      -> Conv3x3 + BN + ReLU + MaxPool(1x8)
      -> Conv3x3 + BN + ReLU + MaxPool(1x8)
      -> Conv3x3 + BN + ReLU + MaxPool(1x4)
      -> Reshape to (B, T, 128)
      -> BiLSTM(64 units each direction)  -> (B, T, 128)
      -> FF 429 -> ReLU -> FF 429         -> (B, T, 429 logits)
    """
    def __init__(self, num_mics: int = 6, num_classes: int = 429):
        super().__init__()
        self.c1 = ConvBlock(num_mics, 64, pool_kernel=(1, 8))
        self.c2 = ConvBlock(64, 64,       pool_kernel=(1, 8))
        self.c3 = ConvBlock(64, 64,       pool_kernel=(1, 4))

        # With F=513 and pools (8, 8, 4), width becomes 2.
        # Channels after conv3 = 64, so 64 * 2 = 128 features per time step.
        self.expected_feat_per_timestep = 128

        self.rnn = nn.LSTM(
            input_size=128, hidden_size=64,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.ff1 = nn.Linear(128, num_classes)
        self.act = nn.ReLU(inplace=True)
        self.ff2 = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        """
        x: Tensor shaped (B, T, F, C) = (batch, time=25, freq=513, channels=6)
        returns logits of shape (B, T, 429)
        """
        # -> (B, C, T, F) for Conv2d
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.c1(x)  # (B, 64, T=25, F≈64)
        x = self.c2(x)  # (B, 64, T=25, F=8)
        x = self.c3(x)  # (B, 64, T=25, F=2)

        B, C, T, W = x.shape
        # Reshape to (B, T, C*W) = (B, 25, 128)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * W)

        # Safety check: match the figure’s 128-dim step features
        if x.size(-1) != self.expected_feat_per_timestep:
            raise ValueError(
                f"Expected 128 features per timestep after convs, got {x.size(-1)}. "
                "Make sure input F=513 (so after pooling 8x,8x,4x → width=2)."
            )

        x, _ = self.rnn(x)          # (B, T, 128)
        x = self.act(self.ff1(x))   # (B, T, 429)
        logits = self.ff2(x)        # (B, T, 429)
        return logits


# ----------------------------
# Minimal training loop (example)
# ----------------------------
class DummyDoADataset(Dataset):
    """
    Replace this with your real dataset.
    Targets are class indices in [0, 428], one per time-step (length 25).
    """
    def __init__(self, n_items=256, T=25, F=513, C=12, n_classes=72):
        super().__init__()
        self.X = torch.randn(n_items, T, F, C)
        self.y = torch.randint(0, n_classes, (n_items, T))

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_one_epoch(model, loader, optimizer, device, num_classes=429):
    model.train()
    ce = nn.CrossEntropyLoss()
    running_loss, running_correct, running_total = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)              # xb: (B,T,F,C), yb: (B,T)
        print(xb.shape)
        print(yb.shape)
        
        logits = model(xb)                                 # (B,T,429)

        B, T, C = logits.shape
        print(logits.view(B * T, C).shape)
        print(yb.view(B * T).shape)




        loss = ce(logits.view(B * T, C), yb.view(B * T))   # CE across all time-steps (all time-steps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * B
        preds = logits.argmax(dim=-1)                      # (B,T)
        running_correct += (preds == yb).sum().item()
        running_total += B * T

    avg_loss = running_loss / len(loader.dataset)
    acc = running_correct / running_total
    return avg_loss, acc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DoAEstimator(num_mics=12, num_classes=72).to(device)

    # Replace with your real dataset & DataLoader
    train_ds = DummyDoADataset(n_items=512)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in range(5):  # demo
        loss, acc = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}: loss={loss:.4f}  token-acc={acc:.3f}")
