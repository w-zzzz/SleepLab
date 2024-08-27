import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import butter, filtfilt, detrend
from .utils import extract_data_subset
from .utils import resample_data


plt.rcParams['figure.figsize'] = (18, 3)

class CWDataProcessor:
    def __init__(self, file_path, sample_rate):
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.data = None
        self.start_datetime = None
        self.end_datetime = None
    
    def load_data(self):
        """
        Load the radar data from a CSV file.
        """
        self.data = pd.read_csv(self.file_path)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], format='%Y%m%d%H%M%S%f')
        self.start_datetime = self.data['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
        print(f"Start time: {self.start_datetime}")
        self.end_datetime = self.data['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
        print(f"End time: {self.end_datetime}")
        print(f"Sample rate: {self.sample_rate} Hz")

    def process_signal(self, data):
        """
        Process the signal by performing arctan demodulation, phase unwrapping,
        detrending, and low-pass filtering.

        Parameters:
        - data: DataFrame containing the in-phase (data_i) and quadrature (data_q) components of the signal.

        Returns:
        - processed_df: DataFrame with timestamp as the index and the processed signal as the 'processed_signal' column.
        """
        # Extract data_i and data_q as floats
        data_i = data['data_i'].astype(float)
        data_q = data['data_q'].astype(float)

        # Perform arctan demodulation
        sig = np.arctan2(data_q, data_i)

        # Unwrap the phase
        sig_unwrapped = np.unwrap(np.angle(sig))

        # Detrend the unwrapped signal
        sig_unwrapped = detrend(sig_unwrapped)

        # Design a low-pass filter
        b, a = butter(4, 5 / (0.5 * self.sample_rate), btype='low')

        # Apply the filter to the detrended signal
        filtered_signal = filtfilt(b, a, sig_unwrapped)

        # Create a DataFrame with the timestamp as the index and the filtered signal as a column
        processed_df = pd.DataFrame(filtered_signal, index=data['timestamp'], columns=['processed_signal'])

        return processed_df

    def plot_processed_data(self, data):
        """
        Plots the processed signal data with time labels for start and end datetimes.

        Parameters:
        - data: DataFrame containing the signal data to be processed and plotted.
        """
        filtered_signal_df = self.process_signal(data)
        filtered_signal = filtered_signal_df['processed_signal']

        plt.figure(figsize=(12, 6))
        plt.plot(filtered_signal, label='Unwrapped Phase')
        plt.xlabel('Time (h:m:s)')
        plt.ylabel('Phase (radians)')
        plt.title('Phase Unwrapping of Radar Signal')
        plt.legend()
        plt.grid(True)

        # Set x-ticks to show start and end time
        # plt.xticks([0, len(filtered_signal) - 1], [data.iloc[0]['timestamp'].strftime('%H:%M:%S'), data.iloc[-1]['timestamp'].strftime('%H:%M:%S')])

        plt.tight_layout()
        plt.show()

    def plot_frequency_analysis(self, data):
        """
        Perform frequency analysis and plot the FFT of the unwrapped signal.

        Parameters:
        - data: DataFrame containing the signal data to be analyzed and plotted.
        """
        filtered_signal_df = self.process_signal(data)
        sig_unwrapped = filtered_signal_df['processed_signal']

        # Perform FFT
        fft_values = np.fft.fft(sig_unwrapped)
        N = len(sig_unwrapped)
        T = 1 / self.sample_rate
        frequencies = np.fft.fftfreq(N, T)
        magnitude = np.abs(fft_values)

        # Select the appropriate frequency range
        if N % 2 == 0:
            freq_range = frequencies[:N // 2]
            magnitude_range = magnitude[:N // 2]
        else:
            freq_range = frequencies[:(N + 1) // 2]
            magnitude_range = magnitude[:(N + 1) // 2]

        plt.figure(figsize=(10, 6))
        plt.plot(freq_range, magnitude_range)
        plt.title('One-sided Frequency Map of Processed Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(0, 5)  # Adjust as needed
        plt.grid(True)
        plt.show()


# Example usage:
if __name__ == "__main__":
    # file_path = "/Users/w.z/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/SleepData/苏州大学附属医院/Radar/radar20240620220948433561.csv"
    file_path = "/opt/data/private/ZhouWenren/SleepLab/cw_radar/radar20240620220948433561.csv"
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
