# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Jean-François Brisson, Spark AI NLP

import math
import sys
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
import scipy.signal as signal

# Configure PyQtGraph global aesthetics for dark-mode telemetry style
pg.setConfigOptions(antialias=True, background="k", foreground="w")


class OES32Dashboard(QMainWindow):

  def __init__(self):
    super().__init__()
    self.setWindowTitle("OES-32 Real-Time Telemetry & Spectral Dashboard")
    self.resize(1200, 800)

    # Simulation & DSP Parameters
    self.fs = 250.0  # 250 Hz sampling rate
    self.num_channels = 32
    self.window_size = 512  # Rolling buffer size for time/frequency plots
    self.time_buffer = np.zeros((self.window_size, self.num_channels))
    self.t_counter = 0.0

    # 1. 50 Hz Notch Filter Setup
    b_notch, a_notch = signal.iirnotch(50.0, 30.0, fs=self.fs)
    self.sos_notch = signal.tf2sos(b_notch, a_notch)
    zi_notch = signal.sosfilt_zi(self.sos_notch)
    self.zi_notch = np.repeat(
        zi_notch[:, :, np.newaxis], self.num_channels, axis=2
    )

    # 2. Bandpass Filter Setup (0.5Hz - 45Hz Butterworth)
    self.sos_bp = signal.butter(
        4, [0.5, 45.0], btype="bandpass", fs=self.fs, output="sos"
    )
    zi_bp = signal.sosfilt_zi(self.sos_bp)
    self.zi_bp = np.repeat(zi_bp[:, :, np.newaxis], self.num_channels, axis=2)

    self.init_ui()

    # UI Refresh Timer (e.g., 40Hz refresh rate = 25ms interval)
    self.timer = QTimer()
    self.timer.timeout.connect(self.update_plots)
    self.timer.start(25)

  def init_ui(self):
    central_widget = QWidget()
    self.setCentralWidget(central_widget)
    main_layout = QVBoxLayout(central_widget)

    # Dashboard Title Header
    title_label = QLabel(
        "OES-32 Bio-Telemetry Suite — Live Waveform & Welch PSD Spectrum"
    )
    title_label.setStyleSheet(
        "font-size: 16px; font-weight: bold; color: #00ffcc; margin-bottom: 5px;"
    )
    main_layout.addWidget(title_label)

    # PyQtGraph Graphics Layout
    self.graphics_layout = pg.GraphicsLayoutWidget()
    main_layout.addWidget(self.graphics_layout)

    # Plot 1: Time Domain Waveform (Channel 0)
    self.plot_time = self.graphics_layout.addPlot(
        row=0, col=0, title="Time Domain Waveform (Channel 0 - Filtered)"
    )
    self.plot_time.setLabel("left", "Amplitude", units="uV")
    self.plot_time.setLabel("bottom", "Buffer Samples")
    self.plot_time.showGrid(x=True, y=True, alpha=0.3)
    self.curve_time = self.plot_time.plot(pen=pg.mkPen("#00ffcc", width=2))

    self.graphics_layout.nextRow()

    # Plot 2: Frequency Domain (Welch PSD Periodogram)
    self.plot_psd = self.graphics_layout.addPlot(
        row=1, col=0, title="Power Spectral Density (Welch Periodogram)"
    )
    self.plot_psd.setLabel("left", "PSD", units="uV^2/Hz")
    self.plot_psd.setLabel("bottom", "Frequency", units="Hz")
    self.plot_psd.setXRange(0, 50, padding=0)
    self.plot_psd.showGrid(x=True, y=True, alpha=0.3)
    self.curve_psd = self.plot_psd.plot(pen=pg.mkPen("#ff00ff", width=2))

  def update_plots(self):
    # Process a batch of incoming samples per tick for smooth animation
    batch_size = 5
    for _ in range(batch_size):
      self.t_counter += 1.0 / self.fs
      t = self.t_counter

      # Simulate 32-channel incoming frame (10Hz Alpha rhythm + 50Hz power line noise)
      raw_sample = []
      for ch in range(self.num_channels):
        alpha_wave = math.sin(2 * math.pi * 10.0 * t + (ch * 0.1)) * 25.0
        power_noise = math.sin(2 * math.pi * 50.0 * t) * 15.0
        raw_sample.append(alpha_wave + power_noise)

      # Pipeline: Raw -> 50Hz Notch -> Bandpass Filter
      x = np.array(raw_sample, dtype=np.float64)[np.newaxis, :]
      y_notch, self.zi_notch = signal.sosfilt(
          self.sos_notch, x, axis=0, zi=self.zi_notch
      )
      y_bp, self.zi_bp = signal.sosfilt(
          self.sos_bp, y_notch, axis=0, zi=self.zi_bp
      )
      filtered_sample = y_bp.squeeze()

      # Push sample into rolling time buffer
      self.time_buffer = np.roll(self.time_buffer, -1, axis=0)
      self.time_buffer[-1, :] = filtered_sample

    # Update Time-Domain Plot (Channel 0)
    self.curve_time.setData(self.time_buffer[:, 0])

    # Compute Welch PSD across the rolling buffer (averaged across channels)
    freqs, psd = signal.welch(self.time_buffer, fs=self.fs, nperseg=256, axis=0)
    mean_psd = np.mean(psd, axis=1)

    # Update Frequency-Domain Plot
    self.curve_psd.setData(freqs, mean_psd)


if __name__ == "__main__":
  app = QApplication(sys.argv)
  dashboard = OES32Dashboard()
  dashboard.show()
  sys.exit(app.exec())
