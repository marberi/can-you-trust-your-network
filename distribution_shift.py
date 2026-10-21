# %% [markdown]
# # Distribution shift: train on low redshift, test on high
#
# The same CNN, trained only on spectra with z < 0.5, then shown spectra with z > 0.8.

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
cat = pd.read_parquet(DATA_DIR / 'catalog.parquet')
spectra = zarr.open(DATA_DIR / 'spectra.zarr', mode='r')[:]   # (N, 2, 7781): flux, weight
print(cat.shape, spectra.shape)

# %% [markdown]
# ## Split by redshift

# %%
flux = torch.tensor(spectra[:, :1]).to(device)
z = torch.tensor(cat.z.values, dtype=torch.float32).to(device)

low, high = z < 0.5, z > 0.8
ds_low = TensorDataset(flux[low], z[low].unsqueeze(1))
n_train = int(0.8 * len(ds_low))
ds_train, ds_test = random_split(ds_low, (n_train, len(ds_low) - n_train),
                                 generator=torch.Generator().manual_seed(42))
ds_shift = TensorDataset(flux[high], z[high].unsqueeze(1))   # never seen in training

# %% [markdown]
# ## Train on z < 0.5 only
# The printed metrics are for held-out low-redshift spectra: they look good, which is the trap.

# %%
from redshift_cnn import RedshiftCNN, train, predict

model = RedshiftCNN(n_channels=1).to(device)
train(model, ds_train, ds_test, epochs=20)

# %% [markdown]
# ## Test on data like the training set, and on new data

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for a, ds, title in [(axes[0], ds_test, 'Test: z < 0.5 (like training)'),
                     (axes[1], ds_shift, 'New data: z > 0.8')]:
    res = predict(model, DataLoader(ds, batch_size=512))
    a.plot(res.true, res.pred, '.', ms=0.5)
    a.plot([0, 1.7], [0, 1.7], 'k-', lw=1)
    a.axhspan(0, 0.5, color='C1', alpha=0.1, label='training range')
    a.set_xlim(0, 1.7)
    a.set_ylim(0, 1.7)
    a.set_title(title, fontsize=14)
    a.set_xlabel('True redshift')
axes[0].set_ylabel('Predicted redshift')
axes[0].legend()

# %%
