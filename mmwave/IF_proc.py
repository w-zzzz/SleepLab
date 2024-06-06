'''
Author: Xuelin Kong
Date: 2024-04-26 15:34:34
LastEditors: Do not edit
LastEditTime: 2024-04-26 15:58:24
FilePath: \2243dataprocessing\mmwave\IF_proc.py
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter, find_peaks, welch, detrend
from scipy.fftpack import fft
from scipy.interpolate import interp1d

class IFSignalProcessor:
    def __init__(self, iq_data, period, sampling_interval, plot_enabled=False):
        self.iq_data = iq_data
        self.period = period
        self.sampling_interval = sampling_interval
        self.plot_enabled = plot_enabled

        # Pre-compute common parameters
        self.n_proc = len(iq_data)
        self.t = np.arange(0, self.period * self.n_proc, self.period)
        self.fs = 1e3 * self.sampling_interval

    def phase_unwrapping(self):
        angle_data = np.unwrap(np.angle(self.iq_data))
        angle_data = angle_data - np.polyval(np.polyfit(self.t, angle_data, 1), self.t)
        if self.plot_enabled:
            self.plot_phase_unwrapping(angle_data)
        return angle_data

    def plot_phase_unwrapping(self, angle_data):
        plt.subplot(2, 1, 1)
        plt.plot(self.t, angle_data)
        plt.xlabel('t(s)')
        plt.ylabel('Phase (rad)')
        plt.title('Phase Unwrapping Result')

    def fft_of_signal(self, signal):
        n = len(signal)
        fft_signal = np.abs(fft(signal))
        f = np.arange(0, n) * (self.fs / n)
        if self.plot_enabled:
            self.plot_fft(f[:n//2], fft_signal[:n//2])
        return f[:n//2], fft_signal[:n//2]

    def plot_fft(self, frequency, magnitude):
        plt.subplot(2, 1, 2)
        plt.plot(frequency, magnitude)
        plt.xlim([0.05, 3])
        plt.xlabel('Frequency (f/Hz)')
        plt.ylabel('Magnitude')
        plt.title('FFT of Phase Signal')

    def lowpass_filter(self, signal, cutoff=5, order=12):
        d1 = butter(order, cutoff, btype='low', fs=self.fs)
        filtered_signal = filtfilt(d1[0], d1[1], signal)
        if self.plot_enabled:
            self.plot_lowpass(filtered_signal)
        return filtered_signal

    def plot_lowpass(self, filtered_signal):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.t, filtered_signal)
        plt.xlabel('t(s)')
        plt.ylabel('Phase (rad)')
        plt.title('Signal after Lowpass Filtering')

    def smooth_signal(self, signal, window_size=4):
        smoothed_signal = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
        if self.plot_enabled:
            self.plot_smoothing(smoothed_signal)
        return smoothed_signal

    def plot_smoothing(self, smoothed_signal):
        plt.subplot(2, 1, 1)
        plt.plot(self.t[:len(smoothed_signal)], smoothed_signal)
        plt.title('Smoothed Phase Signal')

    def remove_dc_component(self):
        # Remove the DC component from the IQ data
        return detrend(self.iq_data)


    def bandpass_filter(self, signal, lowcut, highcut, order=5):
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal


# Example usage:

# period = 40e-3  # Replace with your actual period value
# sampling_interval = ...  # Replace with your actual sampling interval
# processor = IFSignalProcessor(iq_data, period, sampling_interval, plot_enabled=True)
# phase_data = processor.phase_unwrapping()
# frequency, magnitude = processor.fft_of_signal(phase_data)
# filtered_signal = processor.lowpass_filter(phase_data)
# smoothed_signal = processor.smooth_signal(filtered_signal)
