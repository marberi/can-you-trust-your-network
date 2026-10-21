# %% [markdown]
# # A normalizing flow for redshift: p(z | spectrum)
#
# Same data and split as the CNN notebook; the output head is replaced by a conditional flow.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split

# %%
from huggingface_hub import snapshot_download

# Downloads the dataset once (about 5 GB) and reuses the local copy afterwards.
DATA_DIR = Path(snapshot_download('marberi/desi-dr1-spectra-100k', repo_type='dataset'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Wavelength of each bin. Assumed linear DESI grid; replace with the grid shipped with the data.
WAVE = 3600 + 0.8 * np.arange(7781)

# %% [markdown]
# ## Load the data

# %%
cat = pd.read_parquet(DATA_DIR / 'catalog.parquet').reset_index(drop=True)
spectra = zarr.open(DATA_DIR / 'spectra.zarr', mode='r')[:]   # (N, 2, 7781): flux, weight
print(cat.shape, spectra.shape)

# %% [markdown]
# ## Look at the data

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i in [0, 100]:
    axes[0].plot(WAVE, spectra[i, 0], lw=0.4, label=f'z = {cat.z[i]:.3f}')
axes[0].set_xlabel('Wavelength (Å)')
axes[0].set_ylabel('Normalized flux')
axes[0].legend()
cat.z.hist(bins=100, ax=axes[1])
axes[1].set_xlabel('Redshift')

# %% [markdown]
# ## Train/test split
# Only the flux channel is used. The fixed seed gives the same split in both notebooks.

# %%
flux = torch.tensor(spectra[:, :1])
z = torch.tensor(cat.z.values, dtype=torch.float32)

ds = TensorDataset(flux.to(device), z.unsqueeze(1).to(device))
n_train = int(0.8 * len(ds))
ds_train, ds_test = random_split(ds, (n_train, len(ds) - n_train),
                                 generator=torch.Generator().manual_seed(42))

# %% [markdown]
# ## Train

# %%
from redshift_flow import (RedshiftFlow, train_flow, predict, pit,
                           posterior_summary, get_embeddings)
from redshift_cnn import metrics
from plotting import scatter, plot_examples, pca_comparison, umap_comparison

model = RedshiftFlow(n_channels=1, z_mean=z.mean(), z_std=z.std()).to(device)
train_flow(model, ds_train, ds_test, epochs=100, lr=1e-4)

# %% [markdown]
# ## Predictions (posterior median)

# %%
dl_train = DataLoader(ds_train, batch_size=512)
dl_test = DataLoader(ds_test, batch_size=512)   # not shuffled: row i is ds_test[i]

res = predict(model, dl_test)
nmad, outliers = metrics(res)
print(f'sigma_NMAD {nmad:.4f}  outliers {100 * outliers:.1f}%')

scatter(res, label='flow, posterior median')

# %% [markdown]
# ## Are the error bars honest? (PIT)
# A flat histogram means calibrated posteriors.

# %%
K = {'bins': 100, 'density': True, 'histtype': 'step'}
pit(model, dl_train).hist(label='Train', **K)
pit(model, dl_test).hist(label='Test', **K)
plt.xlabel('PIT', size=14)
plt.legend()

# %% [markdown]
# ## Example posteriors
# Summarize ~3000 test posteriors, then pick examples by width and number of peaks.

# %%
post = posterior_summary(model, dl_test, max_batches=6)

narrow  = post[post.n_peaks == 1].nsmallest(20, 'width68')
broad   = post[post.n_peaks == 1].nlargest(20, 'width68')
twopeak = post[post.n_peaks >= 2].sort_values('width95', ascending=False).head(20)

# %%
idxs = [narrow.index[0], twopeak.index[0], broad.index[0]]
fig = plot_examples(model, ds_test, WAVE, idxs, titles=['Narrow', 'Two peaks', 'Long tail'])

# %% [markdown]
# ## What the network has learned
# The 128-number embedding, projected to 2D, against the same projection of the raw spectra.

# %%
raw, emb, z_emb = get_embeddings(model, dl_test, max_batches=10)   # ~5000 spectra
fig = pca_comparison(raw, emb, z_emb)

# %%
fig = umap_comparison(raw, emb, z_emb)

# %%
plt.show()
