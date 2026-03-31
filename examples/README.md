# barc4dip - Example Notebooks

v2026.03

This folder contains a minimal set of Jupyter notebooks demonstrating the core functionalities of **barc4dip**.

---

## Example 1: Sharpness scan
**File:** `ex_01_sharpness_scan.ipynb`

Shows how to:
- Compute sharpness metrics on a scan of images.
- Compare different focus metrics (e.g. Tenengrad, Laplacian).
- Visualize metric evolution across the scan.
- Identify optimal focus conditions.

---

## Example 2: Speckle field statistics
**File:** `ex_02_speckle_field_statistics.ipynb`

Shows how to:
- Compute speckle metrics (amplitude, grain size, statistics, bandwidth).
- Analyze both full-image and tiled (3×3) statistics.
- Interpret speckle properties in terms of contrast and coherence.
- Visualize results using barc4dip plotting utilities.

---

## Example 3: FFT, PSD, and autocorrelation of a speckle field
**File:** `ex_03_speckle_field_fft_psd_autocorrelation.ipynb`

Shows how to:
- Compute FFT, power spectral density (PSD), and autocorrelation.
- Extract characteristic speckle sizes from autocorrelation.
- Perform radial averaging of spectral quantities.
- Visualize spectral data with cuts and radial profiles.

---

## Example 4: Speckle stack temporal statistics
**File:** `ex_04_speckle_stack_temporal_statistics.ipynb`

Shows how to:
- Analyze time-resolved speckle stacks.
- Track speckle displacements using template or phase correlation.
- Compute temporal statistics (absolute and incremental motion).
- Visualize trajectories and time series of displacement metrics.
