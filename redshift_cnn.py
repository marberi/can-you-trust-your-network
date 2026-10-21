"""1D CNN that predicts redshift from DESI spectra.

Input:  (batch, n_channels, 7781)  normalized flux (and optionally a weight channel)
Output: (batch, 1)                 redshift
"""
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def conv_block(c_in, c_out, kernel=9, pool=4):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=kernel, padding=kernel // 2),
        nn.GroupNorm(4, c_out),
        nn.ReLU(),
        nn.MaxPool1d(pool),
    )


class RedshiftCNN(nn.Module):
    def __init__(self, n_wave=7781, n_channels=1, n_embed=128):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(n_channels, 16),  # 7781 -> 1945
            conv_block(16, 32),          # 1945 -> 486
            conv_block(32, 64),          # 486  -> 121
            conv_block(64, 64),          # 121  -> 30
        )
        # Flatten instead of global pooling: WHERE a line sits is the redshift,
        # so the network must keep track of position.
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, n_channels, n_wave)).numel()

        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, n_embed),
            nn.ReLU(),
        )
        self.head = nn.Linear(n_embed, 1)

    def features(self, x):
        """The 128-number embedding."""
        return self.embed(self.conv(x))

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def predict(model, dl):
    """True and predicted redshifts, one row per spectrum."""
    model.eval()
    true, pred = [], []
    for x, z in dl:
        pred.append(model(x).squeeze(-1))
        true.append(z.squeeze(-1))
    return pd.DataFrame({'true': torch.cat(true).cpu().numpy(),
                         'pred': torch.cat(pred).cpu().numpy()})


def metrics(res):
    """sigma_NMAD and outlier fraction of (pred - true) / (1 + true)."""
    dz = (res.pred - res.true) / (1 + res.true)
    nmad = 1.4826 * (dz - dz.median()).abs().median()
    outliers = (dz.abs() > 0.15).mean()
    return nmad, outliers


def train(model, ds_train, ds_test, epochs=10, batch_size=128, lr=1e-3):
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=1024)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, z in dl_train:
            opt.zero_grad()
            loss = loss_fn(model(x), z)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)

        nmad, outliers = metrics(predict(model, dl_test))
        print(f"epoch {epoch:3d}  train loss {total / len(dl_train.dataset):.4f}  "
              f"sigma_NMAD {nmad:.4f}  outliers {100 * outliers:.1f}%")
    return model
