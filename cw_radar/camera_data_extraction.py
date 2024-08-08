import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from .utils import extract_data_subset

plt.rcParams['figure.figsize'] = (18, 3)

class CameraDataProcessor:
    def __init__(self, file_path):
        """
        Initialize the CameraDataProcessor class.
        
        Args:
        file_path (str): Path to the CSV file containing camera data.
        """
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """
        Load data from the CSV file and convert timestamps to datetime format.
        """
        self.df = pd.read_csv(self.file_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='%Y%m%d%H%M%S%f')

    def get_rgb_ranges(self):
        """
        Get the range (min and max) of the RGB channels.

        Returns:
        dict: A dictionary containing the min and max values of the RGB channels.
        """
        min_r = self.df['mean_r'].min()
        max_r = self.df['mean_r'].max()

        min_g = self.df['mean_g'].min()
        max_g = self.df['mean_g'].max()

        min_b = self.df['mean_b'].min()
        max_b = self.df['mean_b'].max()

        print(f"Range of r: {min_r} to {max_r}")
        print(f"Range of g: {min_g} to {max_g}")
        print(f"Range of b: {min_b} to {max_b}")

        return {
            'r': (min_r, max_r),
            'g': (min_g, max_g),
            'b': (min_b, max_b)
        }

    def extract_data_subset(self, start_datetime, end_datetime):
        """
        Extract a subset of the data between the given start and end datetimes.
        
        Args:
        start_datetime (datetime): The start datetime for the data subset.
        end_datetime (datetime): The end datetime for the data subset.
        
        Returns:
        pd.DataFrame: The subset of the data within the specified time range.
        """
        mask = (self.df['timestamp'] >= start_datetime) & (self.df['timestamp'] <= end_datetime)
        return self.df.loc[mask]

    def plot_rgb(self, data):
        """
        Plot the green channel of image data with time labels for start and end datetimes.
        
        Args:
        start_datetime (datetime): The start datetime for the plot.
        end_datetime (datetime): The end datetime for the plot.
        """
        green_channel = data['mean_g']

        plt.figure(figsize=(12, 6))
        plt.plot(green_channel, label='Green Channel', color='g')
        plt.xlabel('Time (h:m:s)')
        plt.ylabel('Intensity')
        plt.title('Green Channel Intensity Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Specify the path to the CSV file
    # file_path = '/path/to/your/csv_file.csv'
    file_path = '/Users/w.z/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/SleepData/苏州大学附属医院/Camera/camera20240620220949415042.csv'
    
    # Create an instance of the CameraDataProcessor and load the data
    camera_processor = CameraDataProcessor(file_path)
    camera_processor.load_data()
    
    # Display the first few rows of the dataframe
    print(camera_processor.df.head())
    
    # Print the range of values for the RGB channels
    camera_processor.get_rgb_ranges()
    
    # Define start and end datetime objects
    start_datetime = datetime.strptime('20240620221948', '%Y%m%d%H%M%S')
    end_datetime = datetime.strptime('20240620222049', '%Y%m%d%H%M%S')
    
    # Extract the data subset
    df_subset = extract_data_subset(camera_processor.df, start_datetime, end_datetime)
    
    # Display the first few rows of the subset
    print(df_subset.head())
    
    # Plot the green channel intensity over time
    camera_processor.plot_rgb(df_subset)