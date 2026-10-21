"""Conditional normalizing flow for redshift: p(z | spectrum).

Same CNN encoder as redshift_cnn.py, but the single-number output head is
replaced by a zuko flow conditioned on the 128-number embedding.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zuko
from scipy.signal import find_peaks
from torch.utils.data import DataLoader

from redshift_cnn import RedshiftCNN, metrics


class RedshiftFlow(nn.Module):
    def __init__(self, n_channels=1, n_embed=128, z_mean=0.0, z_std=1.0, encoder=None):
        super().__init__()
        # Reuse a trained CNN if given (its .features gives the embedding).
        self.encoder = encoder or RedshiftCNN(n_channels=n_channels, n_embed=n_embed)
        self.flow = zuko.flows.NSF(features=1, context=n_embed, transforms=3,
                                   hidden_features=(256, 256))
        # Standardize z: the splines act on a bounded range.
        self.register_buffer("z_mean", torch.tensor(float(z_mean)))
        self.register_buffer("z_std", torch.tensor(float(z_std)))

    def posterior(self, x):
        """Distribution over standardized z, one per spectrum."""
        return self.flow(self.encoder.features(x))

    def log_prob(self, x, z):
        z_s = (z - self.z_mean) / self.z_std
        # log p(z) = log p(z_s) - log z_std  (change of variables)
        return self.posterior(x).log_prob(z_s) - torch.log(self.z_std)

    @torch.no_grad()
    def sample(self, x, n=1000):
        """Samples of z, shape (n, batch, 1)."""
        z_s = self.posterior(x).sample((n,))
        return z_s * self.z_std + self.z_mean


@torch.no_grad()
def predict(model, dl, n_samples=200):
    """True redshift and posterior median, one row per spectrum."""
    model.eval()
    true, pred = [], []
    for x, z in dl:
        s = model.sample(x, n_samples)            # (n_samples, batch, 1)
        pred.append(s.median(dim=0).values.squeeze(-1))
        true.append(z.squeeze(-1))
    return pd.DataFrame({'true': torch.cat(true).cpu().numpy(),
                         'pred': torch.cat(pred).cpu().numpy()})


@torch.no_grad()
def test_nll(model, dl):
    model.eval()
    return torch.cat([-model.log_prob(x, z) for x, z in dl]).mean().item()


def train_flow(model, ds_train, ds_test, epochs=20, batch_size=128, lr=1e-3):
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=512)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, z in dl_train:
            opt.zero_grad()
            loss = -model.log_prob(x, z).mean()      # the only change from regression
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(x)
        sched.step()

        nmad, outliers = metrics(predict(model, dl_test))
        print(f"epoch {epoch:3d}  train NLL {total / len(dl_train.dataset):.3f}  "
              f"test NLL {test_nll(model, dl_test):.3f}  "
              f"sigma_NMAD {nmad:.4f}  outliers {100 * outliers:.1f}%")
    return model


@torch.no_grad()
def pit(model, dl, n_samples=1000):
    """Probability integral transform: fraction of samples below the true z.
    A flat histogram of these values means calibrated posteriors."""
    model.eval()
    out = []
    for x, z in dl:
        s = model.sample(x, n_samples)            # (n_samples, batch, 1)
        out.append((s < z.unsqueeze(0)).float().mean(dim=0).squeeze(-1))
    return pd.Series(torch.cat(out).cpu().numpy())


@torch.no_grad()
def posterior_summary(model, dl, n_samples=1000, bins=60, max_batches=None):
    """One row per spectrum: true z, median, 68% and 95% widths, number of peaks."""
    model.eval()
    rows = []
    for k, (x, z) in enumerate(dl):
        if max_batches is not None and k >= max_batches:
            break
        s = model.sample(x, n_samples).squeeze(-1).cpu().numpy()   # (n_samples, batch)
        for sj, zj in zip(s.T, z.squeeze(-1).tolist()):
            lo95, lo68, med, hi68, hi95 = np.percentile(sj, [2.5, 16, 50, 84, 97.5])
            h, _ = np.histogram(sj, bins=bins)
            h = np.convolve(h, [0.25, 0.5, 0.25], mode='same')
            peaks, _ = find_peaks(h, prominence=0.15 * h.max())
            rows.append({'true': zj, 'pred': med,
                         'width68': (hi68 - lo68) / 2 / (1 + med),
                         'width95': (hi95 - lo95) / 2 / (1 + med),
                         'n_peaks': len(peaks)})
    return pd.DataFrame(rows)


@torch.no_grad()
def get_embeddings(model, dl, max_batches=None):
    """Raw flux, 128-number embedding and true z for the spectra in dl."""
    model.eval()
    raw, emb, zs = [], [], []
    for k, (x, z) in enumerate(dl):
        if max_batches is not None and k >= max_batches:
            break
        raw.append(x[:, 0].cpu().numpy())
        emb.append(model.encoder.features(x).cpu().numpy())
        zs.append(z.squeeze(-1).cpu().numpy())
    return np.concatenate(raw), np.concatenate(emb), np.concatenate(zs)
