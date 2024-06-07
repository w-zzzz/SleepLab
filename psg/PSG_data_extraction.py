'''
Author: Wenren Zhou
Date: 2024-04-25 13:50:14
LastEditors: Do not edit
LastEditTime: 2024-04-25 20:35:51
FilePath: \2243dataprocessing\psg\PSG_data_extraction.py
'''
import mne
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class PSGDataProcessor:
    def __init__(self, psg_file):
        """
        Initialize the PSGDataProcessor class with the path to the PSG data file.
        
        Args:
        psg_file (str): Path to the PSG file.
        """
        # Load the raw data from EDF file
        self.data = mne.io.read_raw_edf(psg_file, preload=True)
        self.raw_data = self.data.get_data()
        self.sampling_rate = self.data.info['sfreq']
        
        # Adjusting for potential changes in the 'meas_date' type
        if isinstance(self.data.info['meas_date'], tuple):
            # If 'meas_date' is a tuple, convert to datetime object
            meas_date = datetime.fromtimestamp(self.data.info['meas_date'][0])
        else:
            # If 'meas_date' is already a datetime object
            meas_date = self.data.info['meas_date']
        
        self.start_datetime = meas_date.replace(tzinfo=None) # Make timestamp2 offset-naive by removing the timezone information

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
        # Calculate start and end indices based on timestamps and sampling rate
        print(start_datetime, self.start_datetime)
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
            data_array = np.array(self.data[data_type][0][0])
            extracted_data[data_type] = data_array[start_idx:end_idx]
        
        return extracted_data
    
    
    def plot_data(self, data, data_type, sampling_rate):
        """
        Plot the ECG data using Matplotlib.
        
        Args:
        ecg_data (numpy.ndarray): The ECG data to plot.
        sampling_rate (int): The sampling rate of the data.
        """
        # Create time axis in seconds
        time_axis = np.linspace(0, len(data) / sampling_rate, len(data))

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, data, label=data_type)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title(data_type)
        plt.legend()
        plt.grid(True)
        plt.show()
        
# Example usage:
psg_file_path = "../../PSG_Data/sub2/sub2_yuanshishuju.edf"
processor = PSGDataProcessor(psg_file_path)

# Define the start and end datetime for the data segment you want to extract
start_datetime = datetime(2024, 1, 11, 14, 29, 53)
end_datetime = datetime(2024, 1, 11, 14, 30, 23)

# Extract ECG and EEG data between the specified timestamps
data_types = ['ECG', 'Thor']
extracted_data = processor.extract_segment_by_timestamp(start_datetime, end_datetime, data_types)
processor.plot_data(extracted_data['ECG'],'ECG', processor.sampling_rate)
processor.plot_data(extracted_data['Thor'],'Thor', processor.sampling_rate)