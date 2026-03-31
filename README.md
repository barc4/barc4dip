# barc4dip

**barc4dip** is a Python package for digital image processing focused on scientific images and beam diagnostics.

It provides tools for I/O, preprocessing, signal analysis, plotting, and compact reporting, with emphasis on **speckle** and **sharpness** metrics for 2D images and stacks.

---

## Features

- Scientific image I/O (`TIFF`, `EDF`, `HDF5`)
- Image statistics and distribution moments  
- **Speckle** and **sharpness** metrics (images and stacks)
- Preprocessing: flat-field correction, PSF deconvolution
- Signal processing: FFT, PSD, correlation, translation tracking
- Plotting: images, tiles, stack metrics, displacements
- Markdown **logbook reports**
- CLI for speckle analysis

---

## Installation

From PyPI:

```bash
pip install barc4dip
```

From source (editable):

```bash
git clone https://github.com/barc4/barc4dip.git
cd barc4dip
pip install -e .
```

With Jupyter support (examples):

```bash
pip install -e .[examples]
``` 

---

## Modules

- `barc4dip.io` – image I/O  
- `barc4dip.preprocessing` – corrections, deconvolution  
- `barc4dip.metrics` – speckle, sharpness, statistics  
- `barc4dip.signal` – FFT, PSD, correlation, tracking  
- `barc4dip.plotting` – visualization  
- `barc4dip.report` – Markdown summaries  
- `geometry` / `maths` / `utils` – support tools  

---

## Quick example

```python
import barc4dip as dip

img = dip.read_image("image.tif")
stats = dip.sharpness_stats(img)
report = dip.logbook_report(stats)
```

---

## CLI

```bash
barc4dip-speckles -s speckles.tif -o report.md
```

---

## Status

Active development. Documentation and examples are expanding.

---

[![PyPI](https://img.shields.io/pypi/v/barc4dip.svg)](https://pypi.org/project/barc4dip/)
[![License: CeCILL-2.1](https://img.shields.io/badge/license-CeCILL--2.1-blue.svg)](https://opensource.org/licenses/CECILL-2.1)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19347315.svg)](https://doi.org/10.5281/zenodo.19347315)
