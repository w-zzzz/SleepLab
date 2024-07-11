import pandas as pd
import datetime

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
