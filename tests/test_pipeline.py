"""Tests for the dashboard's GUI-free signal path (all inputs SYNTHETIC)."""

import os

import numpy as np
import pytest

from oes32_pipeline import OES32FilterPipeline, mean_welch_psd, synthetic_sample

FS = 250.0


def _run(n=2000):
  pipe = OES32FilterPipeline(32, FS)
  return np.asarray([pipe.process(synthetic_sample(i / FS)) for i in range(n)])


def test_synthetic_sample_shape_and_determinism():
  a, b = synthetic_sample(0.37), synthetic_sample(0.37)
  assert a.shape == (32,)
  np.testing.assert_array_equal(a, b)


def test_filter_removes_50hz_and_keeps_10hz():
  out = _run()[1000:]
  freqs, psd = mean_welch_psd(out, fs=FS)
  p10 = psd[np.argmin(np.abs(freqs - 10.0))]
  p50 = psd[np.argmin(np.abs(freqs - 50.0))]
  assert p10 > 1e3 * p50
  assert 20.0 < np.max(np.abs(out[:, 0])) < 30.0  # 25 a.u. tone survives


def test_psd_shape():
  freqs, psd = mean_welch_psd(np.zeros((512, 32)), fs=FS)
  assert freqs.shape == psd.shape == (129,)
  assert freqs[-1] == pytest.approx(FS / 2)


def test_validation():
  with pytest.raises(ValueError):
    OES32FilterPipeline(0, FS)
  with pytest.raises(ValueError):
    OES32FilterPipeline(32, 90.0)
  with pytest.raises(ValueError):
    OES32FilterPipeline(32, FS).process(np.zeros(31))


def test_dashboard_starts_offscreen():
  pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
  pytest.importorskip("pyqtgraph", exc_type=ImportError)
  os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
  from PyQt6.QtWidgets import QApplication

  import oes32_dashboard

  app = QApplication.instance() or QApplication([])
  window = oes32_dashboard.OES32Dashboard()
  window.timer.stop()
  for _ in range(3):
    window.update_plots()
  assert np.all(np.isfinite(window.time_buffer))
  window.close()
  app.processEvents()
