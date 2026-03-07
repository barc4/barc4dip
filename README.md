# barc4dip

**barc4dip** is a Python package for digital image processing, with a focus on scientific images and beam diagnostics.

It provides utilities for image I/O, preprocessing, signal analysis, plotting, and compact reporting, with current emphasis on **speckle** and **sharpness** metrics for 2D images and image stacks.

---

## Features

- Read and write scientific images (`TIFF`, `EDF`, `HDF5`)
- Compute image statistics and distribution moments
- Compute **speckle metrics** on single images and stacks
- Compute **sharpness metrics** on single images and stacks
- Apply basic preprocessing tools such as flat-field correction and PSF deconvolution
- Perform FFT / PSD / correlation / translation tracking operations
- Plot images, histograms, tile maps, stack metrics, and displacements
- Export compact **Markdown logbook reports**
- Run a small CLI for speckle-image reporting

---

## Installation

Local installation from source:

```bash
git clone https://github.com/barc4/barc4dip.git
cd barc4dip
pip install -e .
```

`pip install barc4dip` is planned, but not available yet.

---

## Main modules

- `barc4dip.io` – image reading and writing
- `barc4dip.preprocessing` – flat-field correction, deconvolution
- `barc4dip.metrics` – speckle, sharpness, and distribution metrics
- `barc4dip.signal` – FFT, PSD, correlation, and tracking helpers
- `barc4dip.plotting` – plotting API for images and stack metrics
- `barc4dip.report` – Markdown logbook summaries
- `barc4dip.geometry` / `maths` / `utils` – supporting tools

---

## Quick example

```python
import barc4dip as dip

img = dip.read_image("image.tif")
stats = dip.sharpness_stats(img)
report = dip.logbook_report(stats)

print(stats["full"])
print(report)
```

For stacks:

```python
import barc4dip as dip

stack = dip.read_image("scan.h5")
stats = dip.sharpness_stack_stats(stack)
```

---

## Command line

A small CLI is currently available for speckle-image analysis:

```bash
barc4dip-speckles -s speckles.tif -o report.md
```

Optional flat and dark images can also be provided.

---

## Status

This project is under active development.

- PyPI distribution: coming soon
- More examples: coming soon
- Full documentation: coming soon

The repository already contains early notebooks and generated report examples that illustrate the intended workflow.

---

## License

CeCILL-2.1
