def segment_and_label_as_dataframe(
    df, 
    segment_sec=10, 
    freq_hz=64, 
    label_type='stage', 
    stage_list=None,
    event_list=None,
    input_signals=['breath', 'ECG', 'Thor', 'Abdo', 'SpO2']
):
    """
    Segments the given DataFrame in steps of segment_size and stores each segment
    as a row in the resulting DataFrame. Each column contains a 1D array corresponding
    to that segment of the signal. The last column is the label (stage or event).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to be segmented.
    segment_sec : int, optional
        The duration of each segment in seconds (default=10).
    freq_hz : int, optional
        The frequency of the data in Hz (default=64).
    label_type : str, optional
        The type of label to attach ('stage' for sleep stage, 'event' for sleep event).
        (default='stage')
    input_signals : list, optional
        The list of columns in `df` corresponding to signals. (default=['processed_signal', 'ECG', 'Thor', 'Abdo', 'SpO2'])
    
    Returns:
    --------
    segmented_df : pandas.DataFrame
        A DataFrame where each row corresponds to one segment:
            - Each input signal column will contain a 1D numpy array of length `segment_size`.
            - The last column, 'label', contains the assigned label for that segment.
        The shape of the output DataFrame is
        [num_segments, len(input_signals) + 1].
    """
    
    # Number of data points per segment
    segment_size = segment_sec * freq_hz
    
    # Define columns for sleep stages and events
    sleep_stages = stage_list
    sleep_events = event_list
    
    data_rows = []
    
    # Loop over the DataFrame in steps of 'segment_size'
    for start_idx in range(0, len(df), segment_size):
        segment = df.iloc[start_idx : start_idx + segment_size]
        
        # If the segment is smaller than segment_size (e.g., the last chunk), skip it
        if len(segment) < segment_size:
            break
        
        # Prepare a row where each column is the segment for that signal
        row_data = []
        for signal_col in input_signals:
            # Store the entire 1D array of the signal chunk
            row_data.append(segment[signal_col].values)
        
        # Determine the label
        if label_type == 'stage':
            # Sum each stage column, pick the one with highest sum (dominant stage)
            sleep_stage_counts = segment[sleep_stages].sum()
            dominant_sleep_stage = (
                sleep_stage_counts.idxmax() 
                if sleep_stage_counts.max() > 0 
                else 'No Stage'
            )
            label = dominant_sleep_stage
        
        elif label_type == 'event':
            # Sum each event column, pick the one with highest sum (dominant event)
            sleep_event_counts = segment[sleep_events].sum()
            dominant_sleep_event = (
                sleep_event_counts.idxmax() 
                if sleep_event_counts.max() > 0 
                else 'No Event'
            )
            label = dominant_sleep_event
        
        else:
            raise ValueError("Invalid label_type. Expected 'stage' or 'event'.")
        
        # Append the label to the row data
        row_data.append(label)
        
        # Finally, add this row to our list of segmented data
        data_rows.append(row_data)
    
    # Create column names: each input signal plus the final 'label'
    columns = input_signals + ['label']
    
    # Build a DataFrame: shape => [num_segments, len(input_signals) + 1]
    segmented_df = pd.DataFrame(data_rows, columns=columns)
    
    return segmented_df

def segment_and_label_as_nd_cubes(df, segment_sec=10, freq_hz=64, label_type='stage', input_signals=['processed_signal', 'ECG', 'Thor', 'Abdo', 'SpO2'], stage_list=None, event_list=None):
    """
    Segments the given DataFrame into n-dimensional cubes for each segment and assigns labels based on label_type.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data to be segmented.
    - segment_sec (int): The duration of each segment in seconds. Default is 10.
    - freq_hz (int): The frequency of the data in Hz. Default is 64.
    - label_type (str): The type of label to attach ('stage' for sleep stage, 'event' for sleep event). Default is 'stage'.
    
    Returns:
    - X (numpy.ndarray): A 3D array (n-dimensional cube) representing the data for each segment.
    - y (numpy.ndarray): An array of labels for each segment.
    """
    # Number of data points per segment
    segment_size = segment_sec * freq_hz
    
    # Define columns for sleep stages and sleep events
    sleep_stages = stage_list
    sleep_events = event_list
    
    # Initialize lists to store the segmented data and labels
    X = []
    y = []

    # Loop over the DataFrame in chunks of segment_size
    for start in range(0, len(df), segment_size):
        # Get current segment
        segment = df.iloc[start:start + segment_size]
        
        # Check if segment size is not less than required (especially for the last segment)
        if len(segment) < segment_size:
            continue
        
        # Create a 2D array (segment_size x number_of_features) for the current segment
        segment_array = segment[input_signals].values
        
        # Append the array to X
        X.append(segment_array)
        
        # Determine the label based on label_type
        if label_type == 'stage':
            # Determine the dominant sleep stage (most frequent one)
            sleep_stage_counts = segment[sleep_stages].sum()
            dominant_sleep_stage = sleep_stage_counts.idxmax() if sleep_stage_counts.max() > 0 else 'No Stage'
            y.append(dominant_sleep_stage)
        elif label_type == 'event':
            # Determine the dominant sleep event (most frequent one)
            sleep_event_counts = segment[sleep_events].sum()
            dominant_sleep_event = sleep_event_counts.idxmax() if sleep_event_counts.max() > 0 else 'No Event'
            y.append(dominant_sleep_event)
        else:
            raise ValueError("Invalid label_type. Expected 'stage' or 'event'.")
    
    # Convert X and y to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y
