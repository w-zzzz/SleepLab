from mmwave.IF_proc import IFSignalProcessor
from trial_utils.radar_data_extraction import RadarDataProcessor
from trial_utils.radar_info_generation_by_trial import trial_info_read,csv_generation
import numpy as np
import os
import pandas as pd
import scipy
from datetime import datetime
import matplotlib.pyplot as plt
import argparse


Args0 = {
    'f0':77,                    # Hz
    'ADCStarttime': 6,    # us
    'slope': 29.982,            # MHz/us
    'idle_time': 100,       # us
    'ramp_end_time': 60,    # us
    'Nadc': 256,             # samples per chirp
    'sample_rate': 10000,    # samples per second (ksps)
    'Rx': 4,                      # Number of RX channels
    'Tx': 1,                      # Number of TX channels
    'Nchirp': 128,                # Number of chirps per frame
    'Period': 40            # ms
}# All transferred to s

Args1 = {
    'f0':77,                    # Hz
    'ADCStarttime': 6,    # us
    'slope': 71.599,            # MHz/us
    'idle_time': 7,       # us
    'ramp_end_time': 55.84,    # us
    'Nadc': 256,             # samples per chirp
    'sample_rate': 5000,    # samples per second (ksps)
    'Rx': 4,                      # Number of RX channels
    'Tx': 1,                      # Number of TX channels
    'Nchirp': 32,                # Number of chirps per frame
    'Period': 30,            # ms
}# All transferred to s


def process_radar_data(args):
    # Display the received arguments (For demonstration)
    print("Received the following configuration for radar data processing:")
    if isinstance(args, dict):
        for key, value in args.items():
            print(f"{key}: {value}")
    else:
        for key, value in vars(args).items():
            print(f"{key}: {value}")
    # You can add the radar data processing logic here using the args
    print("Processing radar data...")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Process radar data with given parameters.')
    Args=Args1
    # Add arguments with default values
    parser.add_argument('--f0', type=float, default=Args['f0'], help='Frequency in Hz (default: 77 Hz)')
    parser.add_argument('--ADCStarttime', type=float, default=Args['ADCStarttime'], help='ADC start time in microseconds (default: 6 us)')
    parser.add_argument('--slope', type=float, default=Args['slope'], help='Slope in MHz/us (default: 29.982 MHz/us)')
    parser.add_argument('--idle_time', type=float, default=Args['idle_time'], help='Idle time in microseconds (default: 100 us)')
    parser.add_argument('--ramp_end_time', type=float, default=Args['ramp_end_time'], help='Ramp end time in microseconds (default: 60 us)')
    parser.add_argument('--Nadc', type=int, default=Args['Nadc'], help='Samples per chirp (default: 256 samples per chirp)')
    parser.add_argument('--sample_rate', type=int, default=Args['sample_rate'], help='Sample rate in ksps (default: 10000 ksps)')
    parser.add_argument('--Rx', type=int, default=Args['Rx'], help='Number of RX channels (default: 4)')
    parser.add_argument('--Tx', type=int, default=Args['Tx'], help='Number of TX channels (default: 1)')
    parser.add_argument('--Nchirp', type=int, default=Args['Nchirp'], help='Number of chirps per frame (default: 128 chirps)')
    parser.add_argument('--Period', type=float, default=Args['Period'], help='Period in ms (default: 40 ms)')
    parser.add_argument('--bin_path', type=str, default='../2024-04-09/', help='path for data (default: "../2024-01-11/trial1")')
    parser.add_argument('--csv_path', type=str, default='../2024-04-09/bin_files_time_info.csv', help='CSV path of datainfo (default: "../2024-01-11/trial1/bin_files_time_info.csv")')
    parser.add_argument('--output_path', type=str, default='../2024-04-09/', help='Output path for data (default: "../2024-01-11/trial1")')

    # Parse the arguments
    args = parser.parse_args()

    # Generate csv file
    if not os.path.exists(args.csv_path):
        capture_start,capture_end,total_duration,bin_file_count=trial_info_read(args.csv_path)
        csv_generation(args.bin_path,capture_start,capture_end,total_duration,bin_file_count)

    # Call the function to process radar data
    process_radar_data(args)
    
    processor = RadarDataProcessor(args,csv_path=args.csv_path,output_path=args.output_path)
    start_time = datetime(2024, 4, 30, 3, 30, 5)
    end_time = datetime(2024, 4, 30, 3, 30, 35)
    print('start time:',start_time,'\n','end time:',end_time)
    radar_rawdata = processor.extract_data_by_timestamp(start_time, end_time)
    print(radar_rawdata.shape)
    Idata,Qdata=processor.target_detection(raw_data=radar_rawdata)
    print(Idata.shape[0])
    total_seconds = (end_time-start_time).total_seconds()
    time_interval = total_seconds / float(Idata.shape[0])
    time_list = np.arange(0, total_seconds, time_interval)
    t = time_list[:Idata.shape[0]]
    plt.subplot(211)
    plt.plot(t,Idata)
    plt.subplot(212)
    plt.plot(t,Qdata)


    iq_data=Idata + 1j*Qdata
    IFprocessor = IFSignalProcessor(iq_data, period=30e-3, sampling_interval=0.03, plot_enabled=True)
    
    phase_data = IFprocessor.phase_unwrapping()
    frequency, magnitude = IFprocessor.fft_of_signal(phase_data)
    filtered_signal = IFprocessor.lowpass_filter(phase_data)
    smoothed_signal = IFprocessor.smooth_signal(filtered_signal)
    plt.show()   