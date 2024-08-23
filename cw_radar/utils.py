import pandas as pd
import datetime
import numpy as np
from scipy.signal import resample_poly

def extract_data_subset(data, start_datetime, end_datetime):
    """
    Extracts a subset of data from a DataFrame given the start and end datetimes.
        
    Args:
    data: DataFrame containing the data.
    start_datetime (datetime): The start datetime for the data subset.
    end_datetime (datetime): The end datetime for the data subset.
    
    Returns:
    pd.DataFrame: A DataFrame containing the subset of data between the specified start and end datetimes.
    """
    data_subset = data[(data['timestamp'] >= start_datetime) & (data['timestamp'] <= end_datetime)]
    return data_subset

def resample_data(data, original_freq, target_freq):
    """
    Resamples the signal data to the target frequency.
    
    Parameters:
    - data: DataFrame or NumPy array containing the data. If DataFrame, it should have signal channels and a datetime index ('Time').
    - original_freq: The original sampling frequency (Hz).
    - target_freq: The target sampling frequency (Hz).
    
    Returns:
    - Resampled data: DataFrame or NumPy array, depending on input type.
    """
    # Determine resampling factor
    if original_freq == target_freq:
        return data  # No resampling needed

    resample_factor = original_freq / target_freq

    if isinstance(data, pd.DataFrame):
        # Resampling for DataFrame
        if original_freq > target_freq:
            # Downsampling
            resampled_data = data.apply(lambda x: resample_poly(x, up=1, down=int(resample_factor)))
        else:
            # Upsampling
            resampled_data = data.apply(lambda x: resample_poly(x, up=int(target_freq/original_freq), down=1))
        
        # # Resample the 'Time' index
        # new_time_index = pd.date_range(start=data.index[0], periods=len(resampled_data), freq=f'{1000/target_freq}ms')
        # resampled_data.index = new_time_index
        
        # Resample the 'Time' index using interpolation
        resampled_data.index = pd.date_range(start=data.index[0], end=data.index[-1], periods=len(resampled_data))


        return resampled_data

    elif isinstance(data, np.ndarray):
        # Resampling for NumPy array
        if original_freq > target_freq:
            # Downsampling
            resampled_array = resample_poly(data, up=1, down=int(resample_factor))
        else:
            # Upsampling
            resampled_array = resample_poly(data, up=int(target_freq/original_freq), down=1)

        return resampled_array

    else:
        raise ValueError("Input data must be a pandas DataFrame or a NumPy array.")

# Example usage with DataFrame
# df_resampled = resample_psg_data(psg_date_segment, original_freq=1024, target_freq=64)

# Example usage with NumPy array
# ecg_resampled = resample_psg_data(ecg_data, original_freq=1024, target_freq=64)

