
def extract_timestamps_from_csv(csv_path):
    # Read start and end times from the CSV file
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        start_time_str = [line for line in lines if 'Capture start time' in line][0].split(' - ')[1].strip()
        end_time_str = [line for line in lines if 'Capture end time' in line][0].split(' - ')[1].strip()
        start_time = datetime.strptime(start_time_str, '%a %b %d %H:%M:%S %Y')
        end_time = datetime.strptime(end_time_str, '%a %b %d %H:%M:%S %Y')
    return start_time, end_time

def one_hot_encode(df_psg, df_xml, encode_type='stage'):
    """
    One-hot encode events or sleep stages and integrate into the PSG DataFrame.

    Parameters:
    - df_psg: DataFrame containing the PSG data.
    - df_xml: XMLProcessor object containing event and sleep stage data.
    - encode_type: str, either 'stage' or 'event' to specify what to encode.

    Returns:
    - DataFrame: PSG DataFrame with one-hot encoded columns integrated.
    """
    if encode_type == 'stage':
        df_data = df_xml.sleep_stages
        time_window = 30
        unique_items = df_data['Sleep Stage'].unique()
        time_column = 'Start Time'
    elif encode_type == 'event':
        df_data = df_xml.events
        time_window = 1
        unique_items = df_data['Name'].unique()
        time_column = 'Start'
    else:
        raise ValueError("Invalid encode_type. Must be 'stage' or 'event'.")

    # Create an empty DataFrame with the same index as df_psg
    one_hot_df = pd.DataFrame(index=df_psg.index)
    time_window = timedelta(seconds=time_window)
    
    # Create one-hot encoded columns for each unique item
    for item in unique_items:
        one_hot_df[item] = 0

    # Fill in the one-hot encoded values based on times
    for _, row in df_data.iterrows():
        start_time = row[time_column]
        
        if encode_type == 'stage':
            end_time = start_time + time_window
            item = row['Sleep Stage']
            mask = (df_psg.index >= start_time) & (df_psg.index < end_time)

        else:  # encode_type == 'event'
            end_time = row['End']
            item = row['Name']
            mask = (df_psg.index >= start_time - time_window) & (df_psg.index < end_time + time_window) # label window: 1s before and after the event
        
        one_hot_df.loc[mask, item] = 1

    return one_hot_df

def integrate(df_psg, df_xml):
    # One-hot encode events and sleep stages
    events_one_hot = one_hot_encode(df_psg, df_xml, encode_type='event')
    stages_one_hot = one_hot_encode(df_psg, df_xml, encode_type='stage')

    # Integrate the one-hot encoded DataFrames into the PSG DataFrame
    return pd.concat([df_psg, stages_one_hot, events_one_hot], axis=1)
