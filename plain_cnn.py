# %% [markdown]
# # A CNN that predicts redshift from DESI spectra
#
# Data: 100,000 DESI DR1 spectra (`catalog.parquet`, `spectra.zarr`).

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
from redshift_cnn import RedshiftCNN, train, predict, metrics
from plotting import scatter

model = RedshiftCNN(n_channels=1).to(device)
train(model, ds_train, ds_test, epochs=200, lr=1e-4)

# %% [markdown]
# ## Results

# %%
dl_test = DataLoader(ds_test, batch_size=1024)
res = predict(model, dl_test)
nmad, outliers = metrics(res)
print(f'sigma_NMAD {nmad:.4f}  outliers {100 * outliers:.1f}%')

scatter(res, label='CNN')
