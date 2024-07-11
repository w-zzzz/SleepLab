# main.py

from utils import extract_data_subset
from camera_data_extraction import CameraDataProcessor
from cw_data_extraction import CWDataProcessor
from datetime import datetime

def main():
    # Example usage of CameraDataProcessor
    file_path = '/path/to/your/csv_file.csv'
    # file_path = '/Users/w.z/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/SleepData/苏州大学附属医院/Camera/camera20240620220949415042.csv'
    
    camera_processor = CameraDataProcessor(file_path)
    camera_processor.load_data()
    print(camera_processor.df.head())
    camera_processor.get_rgb_ranges()
    
    start_datetime = datetime.strptime('20240620221948', '%Y%m%d%H%M%S')
    end_datetime = datetime.strptime('20240620222049', '%Y%m%d%H%M%S')
    df_subset = extract_data_subset(camera_processor.df, start_datetime, end_datetime)
    camera_processor.plot_rgb(df_subset)