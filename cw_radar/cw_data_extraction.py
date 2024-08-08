import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import butter, filtfilt, detrend
from .utils import extract_data_subset

plt.rcParams['figure.figsize'] = (18, 3)

class CWDataProcessor:
    def __init__(self, file_path, sample_rate):
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.data = None
    
    def load_data(self):
        """
        Load the radar data from a CSV file.
        """
        self.data = pd.read_csv(self.file_path)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], format='%Y%m%d%H%M%S%f')
        print(f"Start time: {self.data['timestamp'].iloc[0].strftime('%H%M%S')}")
        print(f"Sample rate: {self.sample_rate} Hz")

    def process_signal(self, data):
        """
        Process the signal by performing arctan demodulation, phase unwrapping,
        detrending, and low-pass filtering.

        Parameters:
        - data: DataFrame containing the in-phase (data_i) and quadrature (data_q) components of the signal.

        Returns:
        - filtered_signal: The processed signal.
        """
        data_i = data['data_i'].astype(float)
        data_q = data['data_q'].astype(float)
        sig = np.arctan2(data_q, data_i)
        sig_unwrapped = np.unwrap(np.angle(sig))
        sig_unwrapped = detrend(sig_unwrapped)
        d1 = butter(4, 5 / (0.5 * self.sample_rate), btype='low')
        filtered_signal = filtfilt(d1[0], d1[1], sig_unwrapped)
        return filtered_signal

    def plot_processed_data(self, data):
        """
        Plots the processed signal data with time labels for start and end datetimes.

        Parameters:
        - start_datetime: The start datetime of the data as a string.
        - end_datetime: The end datetime of the data as a string.
        """
        filtered_signal = self.process_signal(data)
        plt.figure(figsize=(12, 6))
        plt.plot(filtered_signal, label='Unwrapped Phase')
        plt.xlabel('Time (h:m:s)')
        plt.ylabel('Phase (radians)')
        plt.title('Phase Unwrapping of Radar Signal')
        plt.legend()
        plt.grid(True)
        plt.xticks([0, len(filtered_signal) - 1], [start_datetime.strftime('%H:%M:%S'), end_datetime.strftime('%H:%M:%S')])
        plt.tight_layout()
        plt.show()

    def plot_frequency_analysis(self, data):
        """
        Perform frequency analysis and plot the FFT of the unwrapped signal.
        """
        filtered_signal = self.process_signal(data)
        sig_unwrapped = filtered_signal
        fft_values = np.fft.fft(sig_unwrapped)
        N = len(sig_unwrapped)
        T = 1 / self.sample_rate
        frequencies = np.fft.fftfreq(N, T)
        magnitude = np.abs(fft_values)

        if N % 2 == 0:
            freq_range = frequencies[:N // 2]
            magnitude_range = magnitude[:N // 2]
        else:
            freq_range = frequencies[:(N + 1) // 2]
            magnitude_range = magnitude[:(N + 1) // 2]

        plt.figure(figsize=(10, 6))
        plt.plot(freq_range, magnitude_range)
        plt.title('One-sided Frequency Map of sig_unwrapped')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(0, 5)
        plt.grid(True)
        plt.show()


# Example usage:
if __name__ == "__main__":
    file_path = "/Users/w.z/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/SleepData/苏州大学附属医院/Radar/radar20240620220948433561.csv"
    sample_rate = 1000

    # Create an instance of the CWDataProcessor
    cw_processor = CWDataProcessor(file_path, sample_rate)

    # Load the data from the specified file
    cw_processor.load_data()
    print(cw_processor.data.head())
    
    # Define start and end datetime objects
    start_datetime = datetime.strptime('20240620221033', '%Y%m%d%H%M%S')
    end_datetime = datetime.strptime('20240620221133', '%Y%m%d%H%M%S')

    # Extract the data subset
    data_subset = extract_data_subset(cw_processor.data, start_datetime, end_datetime)

    # Plot the processed radar data
    cw_processor.plot_processed_data(data_subset)

    # Frequency analysis: FFT plot
    cw_processor.plot_frequency_analysis(data_subset)
