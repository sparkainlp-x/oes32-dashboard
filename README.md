# OES-32 Real-Time Telemetry & Spectral Dashboard

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Status: research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#what-it-is-not)

**Author:** Jean-François Brisson · **Organization:** Spark AI NLP

Desktop application that displays live filtered bio-telemetry (Channel 0 waveform) and Welch power spectral density (0–50 Hz) using the OES-32 SciPy pipeline.

## What it is

- A self-contained PyQt6 + PyQtGraph desktop dashboard.
- Real-time rolling waveform of Channel 0 after the OES-32 filtering pipeline (50 Hz notch → 0.5–45 Hz bandpass).
- Live Welch PSD periodogram (averaged across all 32 channels), 0–50 Hz.
- Simulated 32-channel input: 10 Hz alpha rhythm + 50 Hz power-line noise.
- Dark-mode high-performance rendering at ~40 Hz UI refresh.

## What it is NOT

- Not certified control software, and not a certification artifact.
- Not hardware, field, or medical software.
- Not a claim about physical, biological, or clinical results. All signals are synthetic.
- Not the normative OES-32 residual. That is [oes32-residual](https://github.com/sparkainlp-x/oes32-residual) (ADR-001).

## Prerequisites

```bash
pip install pyqt6 pyqtgraph numpy scipy
```

## Run

```bash
python oes32_dashboard.py
```

## Technical overview

- Sampling rate: 250 Hz
- Rolling buffer: 512 samples
- Filters: 50 Hz IIR notch (Q=30) followed by 4th-order Butterworth bandpass (0.5–45 Hz)
- Visualization: PyQtGraph GraphicsLayoutWidget with two synchronized plots
- Synthetic stimuli only

## License

This software is available under the GNU Affero General Public License v3.0 only (AGPL-3.0-only); see [LICENSE](LICENSE).

Organizations that want to use it in proprietary products or services without AGPL obligations can contact the author about a commercial license via https://sparkainlpx.xyz.

## Citation

Citation metadata is in [CITATION.cff](CITATION.cff).
