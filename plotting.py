"""Plots used in the lecture: prediction scatter, posterior examples, PCA and UMAP."""
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

LINES = {'[OII]': 3727, 'Hβ': 4861, '[OIII]': 5007, 'Hα': 6563}


def scatter(res, label='CNN', ax=None):
    """Predicted against true redshift, with the 0.15 outlier limits."""
    ax = ax or plt.gca()
    ax.plot(res.true, res.pred, '.', ms=0.2)
    zz = np.linspace(0, res.true.max(), 100)
    ax.plot(zz, zz, 'k-', lw=1, label='z_pred = z_true')
    ax.plot(zz, zz + 0.15 * (1 + zz), 'k--', lw=0.8, label='outlier limit')
    ax.plot(zz, zz - 0.15 * (1 + zz), 'k--', lw=0.8)
    ax.set_xlabel('True redshift (DESI pipeline)', fontsize=14)
    ax.set_ylabel(f'Predicted redshift ({label})', fontsize=14)
    ax.legend(loc='upper left')
    return ax


@torch.no_grad()
def plot_examples(model, ds, wave, idxs, titles=None, n_samples=5000):
    """Spectrum (top) and redshift posterior (bottom) for each index in idxs."""
    model.eval()
    n = len(idxs)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 6), squeeze=False)
    for col, i in enumerate(idxs):
        x, z = ds[i]
        s = model.sample(x.unsqueeze(0), n_samples).squeeze().cpu().numpy()
        z_true = z.item()
        lo68, hi68 = np.percentile(s, [16, 84])

        # Spectrum, with expected line positions at the true redshift
        a = axes[0, col]
        a.plot(wave, x[0].cpu().numpy(), lw=0.4, color='0.3')
        for name, lam in LINES.items():
            lam_obs = lam * (1 + z_true)
            if wave[0] < lam_obs < wave[-1]:
                a.axvline(lam_obs, color='C1', lw=0.8, alpha=0.7)
                a.text(lam_obs, 0.95, name, transform=a.get_xaxis_transform(),
                       rotation=90, va='top', ha='right', fontsize=9, color='C1')
        a.set_xlabel('Wavelength (Å)')
        if col == 0:
            a.set_ylabel('Normalized flux')
        if titles:
            a.set_title(titles[col], fontsize=14)

        # Posterior
        a = axes[1, col]
        a.hist(s, bins=100, density=True, color='C0', alpha=0.8)
        a.axvline(z_true, color='k', ls='--', label=f'true z = {z_true:.3f}')
        a.axvspan(lo68, hi68, color='C0', alpha=0.15, label='68% interval')
        a.set_xlabel('Redshift')
        if col == 0:
            a.set_ylabel('p(z | spectrum)')
        a.legend(fontsize=9)
    fig.tight_layout()
    return fig


def _drop_extremes(raw, drop_frac):
    """Mask that leaves out the spectra with the most extreme flux values."""
    peak = np.abs(raw).max(axis=1)
    return peak < np.percentile(peak, 100 * (1 - drop_frac))


def _standardize(emb):
    return (emb - emb.mean(0)) / (emb.std(0) + 1e-8)


def pca_comparison(raw, emb, z, drop_frac=0.01):
    """First two PCA components of the raw spectra and of the embedding."""
    keep = _drop_extremes(raw, drop_frac)
    pca_raw = PCA(n_components=2).fit(raw[keep])
    p_raw = pca_raw.transform(raw)
    emb_s = _standardize(emb)
    pca_emb = PCA(n_components=2).fit(emb_s)
    p_emb = pca_emb.transform(emb_s)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    panels = [(axes[0], p_raw, pca_raw, keep, 'PCA of the raw spectra'),
              (axes[1], p_emb, pca_emb, np.ones(len(z), bool), 'PCA of the learned embedding')]
    for a, p, pca, sel, title in panels:
        sc = a.scatter(p[:, 0], p[:, 1], c=z, s=2, cmap='viridis', vmin=0, vmax=1.7)
        a.set_xlim(*np.percentile(p[sel, 0], [0.5, 99.5]))
        a.set_ylim(*np.percentile(p[sel, 1], [0.5, 99.5]))
        ev = pca.explained_variance_ratio_
        a.set_xlabel(f'PC 1 ({100 * ev[0]:.0f}% of variance)')
        a.set_ylabel(f'PC 2 ({100 * ev[1]:.0f}% of variance)')
        a.set_title(title, fontsize=14)
    fig.colorbar(sc, ax=axes, label='True redshift')
    return fig


def umap_comparison(raw, emb, z, drop_frac=0.01, n_neighbors=15, min_dist=0.1,
                    random_state=42):
    """UMAP of the raw spectra (via 50 PCA components) and of the embedding."""
    import umap

    keep = _drop_extremes(raw, drop_frac)
    raw50 = PCA(n_components=50).fit_transform(raw[keep])
    settings = dict(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    u_raw = umap.UMAP(**settings).fit_transform(raw50)
    u_emb = umap.UMAP(**settings).fit_transform(_standardize(emb))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for a, u, zz, title in [(axes[0], u_raw, z[keep], 'UMAP of the raw spectra'),
                            (axes[1], u_emb, z, 'UMAP of the learned embedding')]:
        sc = a.scatter(u[:, 0], u[:, 1], c=zz, s=2, cmap='viridis', vmin=0, vmax=1.7)
        a.set_title(title, fontsize=14)
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(sc, ax=axes, label='True redshift')
    return fig
