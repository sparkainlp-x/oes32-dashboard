# OES-32 Real-Time Telemetry & Spectral Dashboard

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23307955.svg)](https://doi.org/10.5281/zenodo.23307955)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![CI](https://github.com/sparkainlp-x/oes32-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/sparkainlp-x/oes32-dashboard/actions/workflows/ci.yml)
[![Status: research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#what-it-is-not)

**Author:** Jean-François Brisson · **Organization:** Spark AI NLP

Desktop application that displays a live filtered **synthetic** multichannel signal (Channel 0 waveform) and its Welch power spectral density (0–50 Hz) using the OES-32 SciPy pipeline.

## What it is

- A self-contained PyQt6 + PyQtGraph desktop dashboard.
- Real-time rolling waveform of Channel 0 after the OES-32 filtering pipeline (50 Hz notch → 0.5–45 Hz bandpass).
- Live Welch PSD periodogram (averaged across all 32 channels), 0–50 Hz.
- Simulated 32-channel input: a 10 Hz test tone plus a 50 Hz mains-like tone, in arbitrary units.
- Dark-mode rendering with a 25 ms UI timer (nominal 40 Hz; frame rate not measured).

## What it is NOT

- Not certified control software, and not a certification artifact.
- Not hardware, field, or medical software.
- Not a claim about physical, biological, or clinical results. All signals are synthetic.
- Not the normative OES-32 residual. That is [oes32-residual](https://github.com/sparkainlp-x/oes32-residual) (ADR-001).

## Prerequisites

```bash
pip install -r requirements.txt
```

## Run

```bash
python oes32_dashboard.py
```

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen pytest -q
```

The signal path lives in `oes32_pipeline.py` (no GUI imports). The tests check that the filters strip the 50 Hz tone and keep the 10 Hz tone, and that the outputs are deterministic and validated. A headless smoke test starts the dashboard window. All test inputs are SYNTHETIC.

## Technical overview

- Sampling rate: 250 Hz
- Rolling buffer: 512 samples
- Filters: 50 Hz IIR notch (Q=30) followed by 4th-order Butterworth bandpass (0.5–45 Hz)
- Visualization: PyQtGraph GraphicsLayoutWidget with two synchronized plots
- Synthetic stimuli only; amplitudes are arbitrary units, not physical units

## License

This software is available under the GNU Affero General Public License v3.0 only (AGPL-3.0-only); see [LICENSE](LICENSE).

Organizations that want to use it in proprietary products or services without AGPL obligations can contact the author about a commercial license via https://sparkainlpx.xyz.

## Cite

Concept DOI (all versions): [10.5281/zenodo.23307955](https://doi.org/10.5281/zenodo.23307955). v0.1.0: [10.5281/zenodo.23307956](https://doi.org/10.5281/zenodo.23307956). See [CITATION.cff](CITATION.cff).
