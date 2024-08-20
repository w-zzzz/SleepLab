# main.py

from cw_radar import *
# from utils import extract_data_subset
# from camera_data_extraction import CameraDataProcessor
# from cw_data_extraction import CWDataProcessor
from datetime import datetime

def main():
    # Example usage of CameraDataProcessor
    file_path = '/path/to/your/csv_file.csv'
    # file_path = '/Users/w.z/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/SleepData/苏州大学附属医院/Camera/camera20240620220949415042.csv'
    file_path = "/opt/data/private/ZhouWenren/SleepLab/cw_radar/camera20240620220949415042.csv"
    camera_processor = CameraDataProcessor(file_path)
    camera_processor.load_data()
    print(camera_processor.df.head())
    camera_processor.get_rgb_ranges()
    
    start_datetime = datetime.strptime('20240620221948', '%Y%m%d%H%M%S')
    end_datetime = datetime.strptime('20240620222049', '%Y%m%d%H%M%S')
    df_subset = extract_data_subset(camera_processor.df, start_datetime, end_datetime)
    camera_processor.plot_rgb(df_subset)

    # Example usage of CWDataProcessor
    file_path = '/path/to/your/csv_file.csv'
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
    
if __name__ == "__main__":
    main()
