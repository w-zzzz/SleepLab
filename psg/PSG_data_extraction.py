'''
Author: Wenren Zhou
Date: 2024-04-25 13:50:14
LastEditors: Do not edit
LastEditTime: 2024-07-09 17:38:00
FilePath: \2243dataprocessing\psg\PSG_data_extraction.py
'''
import mne
import pyedflib
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['figure.figsize']=(12, 6)

class PSGDataProcessor:
    def __init__(self, file_path):
        """
        Initialize the PSGDataProcessor class without loading data immediately.

        Args:
        file_path (str): Path to the PSG file.
        """
        self.file_path = file_path
        self.data = None
        self.raw_data = None
        self.edf_file = None
        self.sampling_rate = None
        self.ch_names = None
        self.start_datetime = None
    
    def load_data(self):
        """
        Load PSG data from an EDF file.
        """
        self.data = mne.io.read_raw_edf(self.file_path, preload=False)
        self.raw_data = self.data.get_data()
        self.sampling_rate = self.data.info['sfreq']
        self.ch_names = self.data.ch_names
        self.start_datetime = self.get_datetime_from_info(self.data.info['meas_date'])
        # self.ch_indices = {channel_name: idx for idx, channel_name in enumerate(self.ch_names)}
        
        # Access loaded data attributes
        print(f"Sampling Rate: {self.sampling_rate}")
        print(f"Channel Names: {self.ch_names}")
        print(f"Start Datetime: {self.start_datetime}")
                
    def psg_plot(self):
        self.data.plot()
        
    # def get_channel_index(self, channel_name):
    #     """
    #     Get the index of a specific channel by its name.

    #     Args:
    #     channel_name (str): Name of the channel.

    #     Returns:
    #     int: Index of the channel, or -1 if not found.
    #     """
    #     if not self.ch_indices:
    #         raise ValueError("Data not loaded. Please load the data first.")
        
    #     index = self.ch_indices.get(channel_name)
    #     if index is None:
    #         raise ValueError(f"Channel '{channel_name}' not found.")
    #     print(f"Index of channel '{channel_name}': {index}")
    #     return index

    def retrieve_info(self, info_name = 'file'):
        """
        Print detailed information about the EDF file.

        Args:
        info_name (str): The type of information to retrieve. Options are 'file', 'label', and 'signal'.
        """
        try:
            self.edf_file = pyedflib.EdfReader(self.file_path)
        except Exception as e:
            print(f"Failed to read EDF file: {e}")
            return
            
        if info_name == 'label':
            PSGDataProcessor.print_label_and_freq(list(zip(self.edf_file.getSignalLabels(), self.edf_file.getSampleFrequencies())))
        elif info_name == 'signal':
            PSGDataProcessor.print_sig_headers(self.edf_file.getSignalHeaders())
        else:
            PSGDataProcessor.print_file_header(self.edf_file.getHeader())

        self.edf_file.close()

    @staticmethod
    def print_label_and_freq(sig_freq):
        """
        Print signal labels and their corresponding sampling frequencies.

        Args:
        sig_freq (list of tuples): List containing signal labels and frequencies.
        """
        print("Signal Labels | Sampling Frequencies")
        print("------------------------------------")
        for label, freq in sig_freq:
            print(f"{label} | {freq}")

    @staticmethod
    def print_sig_headers(signal_headers):
        """
        Print all headers for each signal.

        Args:
        signal_headers (list): List containing headers of each signal.
        """
        for i, header in enumerate(signal_headers):
            print(f"Signal {i+1}:")
            print("Field Name | Value")
            print("------------------")
            for field_name, value in header.items():
                print(f"{field_name} | {value}")
            print("\n")

    @staticmethod
    def print_file_header(file_header):
        """
        Print the main header of the EDF file.

        Args:
        file_header (dict): Header information of the EDF file.
        """
        print("Field Name | Value")
        print("------------------")
        for field_name, value in file_header.items():
            print(f"{field_name} | {value}")
    
    @staticmethod
    def get_datetime_from_info(meas_date):
        """
        Convert measurement date to datetime object, handling various formats.

        Args:
        meas_date (tuple or datetime): The measurement date from MNE info.

        Returns:
        datetime: A datetime object.
        """
        if isinstance(meas_date, tuple):
            return datetime.fromtimestamp(meas_date[0]).replace(tzinfo=None)
        return meas_date.replace(tzinfo=None)
    
    def extract_segment_by_timestamp(self, start_datetime, end_datetime, data_types):
        """
        Extract specific types of data within a specified time range defined by timestamps.
        
        Args:
        start_datetime (datetime): Start datetime object.
        end_datetime (datetime): End datetime object.
        data_types (list): List of data types to extract, e.g., ['ECG', 'EEG'].
        
        Returns:
        dict: Dictionary of extracted data arrays keyed by type.
        """
        start_idx = int((start_datetime - self.start_datetime).total_seconds() * self.sampling_rate)
        end_idx = int((end_datetime - self.start_datetime).total_seconds() * self.sampling_rate)
        
        return self.extract_data_indices(start_idx, end_idx, data_types)

    def extract_data_indices(self, start_idx, end_idx, data_types):
        """
        Extract specific types of data within a specified index range.
        
        Args:
        start_idx (int): Start index.
        end_idx (int): End index.
        data_types (list): List of data types to extract, e.g., ['ECG', 'EEG'].
        
        Returns:
        dict: Dictionary of extracted data arrays keyed by type.
        """
        extracted_data = {}
        for data_type in data_types:
            if data_type in self.ch_names:
                data_array = np.array(self.data[data_type][0][0])
                extracted_data[data_type] = data_array[start_idx:end_idx]
            else:
                raise ValueError(f"Data type {data_type} not found in the dataset.")
            
        return extracted_data

    def plot_data(self, data, data_type, sampling_rate):
        """
        Plot the ECG data using Matplotlib.
        
        Args:
        ecg_data (numpy.ndarray): The ECG data to plot.
        sampling_rate (int): The sampling rate of the data.
        """
        time_axis = np.linspace(0, len(data) / sampling_rate, len(data))
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, data, label=data_type)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title(f'{data_type} Data Plot')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def compare_plot(self, data_dict, channel_names, sampling_rate):
        """
        Plot multiple channels data for comparison using subplots.

        Args:
        data_dict (dict): Dictionary containing data arrays for channels.
        channel_names (list of str): List of channel names to plot.
        sampling_rate (int): The sampling rate of the data.
        """
        num_channels = len(channel_names)
        plt.figure(figsize=(12, 6 * num_channels))
        
        for i, channel in enumerate(channel_names):
            if channel in data_dict:
                ax = plt.subplot(num_channels, 1, i + 1)
                time_axis = np.linspace(0, len(data_dict[channel]) / sampling_rate, len(data_dict[channel]))
                ax.plot(time_axis, data_dict[channel], label=channel)
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('Amplitude')
                ax.set_title(f'{channel} Data Plot')
                ax.legend()
                ax.grid(True)
            else:
                print(f"Data for {channel} not found in the provided data dictionary.")

        plt.tight_layout()
        plt.show()

    def ecg_diagram(self, ecg_slice):
        """
        Process and visualize an ECG signal slice with R-peaks.

        Args:
        ecg_slice (np.array): The slice of ECG data to process.
        """
        # Automatically process the (raw) ECG signal
        ecg_signals, ecg_info = nk.ecg_process(ecg_slice, sampling_rate=self.sampling_rate)

        # Plot the processed ECG signal
        nk.ecg_plot(ecg_signals, ecg_info)

        # Extract clean ECG and R-peaks location
        rpeaks = ecg_info["ECG_R_Peaks"]
        cleaned_ecg = ecg_signals["ECG_Clean"]

        # Visualize R-peaks in ECG signal
        plot = nk.events_plot(rpeaks, cleaned_ecg)
        plt.show()
        
        return ecg_signals, ecg_info
        
    def rsp_diagram(self, rsp_slice):
        """
        Process and visualize a respiratory signal slice with peaks.

        Args:
        rsp_slice (np.array): The slice of RSP data to process.
        """
        # Process the respiratory signal
        rsp_signals, rsp_info = nk.rsp_process(rsp_slice, sampling_rate=self.sampling_rate, report="text")

        # Plot the processed RSP signal
        nk.rsp_plot(rsp_signals, rsp_info)

        # Extract clean RSP and R-peaks location
        cleaned_rsp = rsp_signals["RSP_Clean"]
        peaks = rsp_info["RSP_Peaks"]
        throughs = rsp_info["RSP_Troughs"]
        # rate = rsp_info["RSP_Rate"]
        
        # Visualize R-peaks in RSP signal
        plot = nk.events_plot([peaks, throughs], cleaned_rsp)
        plt.show()

        return rsp_signals, rsp_info

        
    def signals_diagram(self, signals):
        """
        Plot the ECG and RSP signals.

        Args:
            signals (dict): A dictionary containing the ECG and RSP signals.

        Returns:
            None
        """
        signals, info = nk.bio_process(ecg=signals['ECG'], rsp=signals['Pleth'], sampling_rate=self.sampling_rate)
        signals[["ECG_Rate", "RSP_Rate"]].plot(subplots=True)
        print(signals.__dict__['_mgr'].items)

        ## For additional signals like EMG and EOG
        # signals, info = nk.bio_process(ecg=signals['ECG'], rsp=signals['Pleth'], emg=signals['EMG_L'], eog=signals['E1-M2'], sampling_rate=self.sampling_rate)
        # signals[["ECG_Rate", "EMG_Amplitude", "EMG_Activity", "EOG_Rate", "RSP_Rate"]].plot(subplots=True)
        # print(signals.__dict__['_mgr'].items)
 
        
# Example usage:
if __name__ == "__main__":
    # Specify the path to the EDF file
    # psg_file_path = "../../PSG_Data/sub2/sub2_yuanshishuju.edf"
    file_path = "/Users/w.z/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/SleepData/PSG_Data/2024-03-07/002yuanshishuju.edf"

    # Create an instance of the PSGDataProcessor
    psg_processor = PSGDataProcessor(file_path)

    # Load the data from the specified file
    psg_processor.load_data()

    # Overview plot of PSG data
    psg_processor.psg_plot()

    # # Get the index of a specific channel
    # channel_name = 'ECG'
    # psg_processor.get_channel_index(channel_name)

    # Print file header information
    psg_processor.retrieve_info('file')

    # Print signal labels
    psg_processor.retrieve_info('label')

    # Print signal details
    psg_processor.retrieve_info('signal')

    # Plot signals over time
    start_datetime = datetime(2024, 3, 7, 22, 10, 00)  # Replace with your actual start datetime
    end_datetime = datetime(2024, 3, 7, 22, 11, 00)  # Replace with your actual end datetime
    data_types = ['ECG', 'Pleth']  # Replace with your actual data types

    print(f"Start Timestamp: {start_datetime}, End Timestamp: {end_datetime}")  # Print the start and end timestamps of the extracted data
    extracted_data = psg_processor.extract_segment_by_timestamp(start_datetime, end_datetime, data_types)
    psg_processor.plot_data(extracted_data['ECG'], 'ECG', psg_processor.sampling_rate)

    # Plot comparison between signals
    extracted_types = list(extracted_data.keys())
    psg_processor.compare_plot(extracted_data, extracted_types, psg_processor.sampling_rate)

    # Plot ECG signal
    ecg_signals, ecg_info = psg_processor.ecg_diagram(extracted_data['ECG'])

    # Plot RSP signal
    rsp_signals, rsp_info = psg_processor.rsp_diagram(extracted_data['Pleth'])

    # Plot multiple PSG signals
    # data_types = ['ECG', 'Pleth', 'EMG_L', 'E1-M2']
    data_types = ['ECG', 'Pleth']

    extracted_data = psg_processor.extract_segment_by_timestamp(start_datetime, end_datetime, data_types)
    psg_processor.signals_diagram(extracted_data)