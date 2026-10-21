# Redshift Prediction from DESI Spectra

Lecture materials for **MasterCosmos BCN**, demonstrating deep learning approaches to estimate galaxy redshift from DESI DR1 spectra.

## Overview

This repository contains code for two approaches to redshift estimation:

1. **CNN Regression** (`plain_cnn.py`) - A 1D convolutional neural network that predicts a single redshift value
2. **Normalizing Flow** (`using_flow.py`) - A conditional normalizing flow that outputs a full posterior distribution p(z | spectrum)

Additionally, `distribution_shift.py` demonstrates how models can fail when tested on data outside their training distribution.

## Data

The code downloads 100,000 DESI DR1 spectra (~5 GB) from Hugging Face:
- `catalog.parquet` - Metadata including pipeline redshifts
- `spectra.zarr` - Normalized flux arrays (7781 wavelength bins)

## Files

| File | Description |
|------|-------------|
| `redshift_cnn.py` | CNN model definition and training loop |
| `redshift_flow.py` | Normalizing flow model with posterior sampling |
| `plain_cnn.py` | Train and evaluate the CNN |
| `using_flow.py` | Train and evaluate the flow, with PIT calibration checks |
| `distribution_shift.py` | Train on low-z, test on high-z to show generalization failure |
| `plotting.py` | Visualization utilities (scatter plots, PCA, UMAP) |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the scripts as Python files or as Jupyter notebooks (they use the percent format):

```bash
python plain_cnn.py
python using_flow.py
python distribution_shift.py
```

Or open them in Jupyter/VS Code as interactive notebooks.

## Requirements

- Python 3.9+
- PyTorch with CUDA support (recommended)
- See `requirements.txt` for full dependencies
