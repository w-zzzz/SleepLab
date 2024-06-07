'''
Author: Xuelin Kong
Date: 2024-04-25 13:49:09
LastEditors: Do not edit
LastEditTime: 2024-04-25 15:05:36
FilePath: \2243dataprocessing\mmwave\trial_utils\radar_info_generation_by_trial.py
'''
import os
import pandas as pd
from datetime import datetime, timedelta
import re

def extract_time(text):
    # Adjust the regex to match the full timestamp format
    match = re.search(r'\- (.*)', text)
    if match:
        return datetime.strptime(match.group(1), '%a %b %d %H:%M:%S %Y')
    else:
        return None

def trial_info_read(csv_file_path):

    df = pd.read_csv(csv_file_path, usecols=[0], dtype={0: str}, header=None)

    # Drop all rows where the first column is NaN
    df = df.dropna(subset=[0])
    df = df.dropna(how='all')

    # Extract capture start time
    capture_start_cell = df.loc[df[0].str.contains("Capture start time", na=False)].values[0]
    capture_start_time = extract_time(capture_start_cell[0])

    # Extract capture end time
    capture_end_cell = df.loc[df[0].str.contains("Capture end time", na=False)].values[0]
    capture_end_time = extract_time(capture_end_cell[0])

    # Extract duration
    duration_cell = df.loc[df[0].str.contains("Duration", na=False)].values[0]
    duration = int(re.search(r'\d+', duration_cell[0]).group())

    print(f"Capture Start Time: {capture_start_time}")
    print(f"Capture End Time: {capture_end_time}")
    print(f"Duration: {duration}")


    # Read only the third column (index 2) and skip rows with NaN in this column
    df2 = pd.read_csv(csv_file_path, usecols=[1],skip_blank_lines=False)
    df2_numeric = pd.to_numeric(df2.iloc[:, 0], errors='coerce')
    bin_file_count = df2_numeric.notna().sum()
    bin_file_count=int(bin_file_count)
    print(f"Number of data bin files: {bin_file_count}")

    return capture_start_time,capture_end_time,duration,bin_file_count



def get_file_size(file_path):
    """ Returns the file size in bytes. """
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    else:
        raise FileNotFoundError("File not found")


def csv_generation(bin_file_directory,capture_start,capture_end,total_duration,bin_file_count):
    # bin_file_directory = '../2024-01-11/trial1'
    # Dictionary to store the file sizes
    file_sizes = {}
    total_data_points = 0

    for file_name in os.listdir(bin_file_directory):
        if file_name.endswith('.bin'):
            file_path = os.path.join(bin_file_directory, file_name)
            file_size = os.path.getsize(file_path)
            # Assuming each sample includes both real and imaginary parts, each 16 bits:
            data_points = file_size // (2 * 2 * 4)  # Calculate number of complex samples (16-bit real + 16-bit imaginary) for 4 Rx
            file_sizes[file_name] = {'file_size': file_size, 'data_points': data_points}
            total_data_points += data_points

    # Initialize a dictionary to hold the start and end times for each bin file
    bin_files_time_info = {}

    # Calculate the start and end times based on the proportion of data points
    previous_end_time = capture_start

    for file_name, info in file_sizes.items():
        data_points = info['data_points']
        file_duration = (data_points / total_data_points) * total_duration
        start_time = previous_end_time
        end_time = start_time + timedelta(seconds=file_duration)
        duration = end_time - start_time  # Calculate duration for each file
        file_name=str(os.path.join(bin_file_directory,file_name))
        bin_files_time_info[file_name] = {
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': duration.total_seconds(),
            'data_points': data_points
        }
        previous_end_time = end_time  # update the previous end time to the current end time

    # Convert the time information dictionary to a DataFrame
    time_info_df = pd.DataFrame.from_dict(bin_files_time_info, orient='index')

    # Save the bin files time information to a new CSV file
    output_path='../2024-04-09/bin_files_time_info.csv'
    time_info_df.to_csv(output_path)
    print(output_path)


# # #example usage
# csv_file_path = '../2024-04-09/adc_data_2024_04_29_Raw_LogFile.csv'
# capture_start,capture_end,total_duration,bin_file_count=trial_info_read(csv_file_path)
# csv_generation('../2024-04-09',capture_start,capture_end,total_duration,bin_file_count)