import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import pandas as pd

def lowpass_filter(fs, sig, highcut, order=4):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = signal.butter(order, high, btype='lowpass')
    filtered_signal = signal.filtfilt(b, a, sig)
    return filtered_signal

def align_to_common_time_grid(psg_resampled, radar_resampled, freq_hz=64):
    """
    Aligns two resampled dataframes to a common time grid.

    Parameters:
    psg_resampled (pd.DataFrame): The first resampled dataframe.
    radar_resampled (pd.DataFrame): The second resampled dataframe.
    freq_hz (int): The frequency for the common time grid in Hz. Default is 64Hz.

    Returns:
    pd.DataFrame, pd.DataFrame: The aligned dataframes.
    """
    # Convert frequency from Hz to milliseconds
    freq_ms = f'{1000 / freq_hz}ms'

    # Define a common time grid (using the minimum and maximum timestamps)
    common_time_index = pd.date_range(
        start=max(psg_resampled.index.min(), radar_resampled.index.min()), 
        end=min(psg_resampled.index.max(), radar_resampled.index.max()), 
        freq=freq_ms  # Adjust frequency as necessary
    )

    # Reindex and interpolate both dataframes to this common time grid
    psg_aligned = psg_resampled.reindex(common_time_index, method='nearest')
    radar_aligned = radar_resampled.reindex(common_time_index, method='nearest')

    return psg_aligned, radar_aligned

def analyze_time_lag(merged_df, events_df, index, sec=50, signal1='Thor', signal2='breath', 
                    sampling_rate=64, max_lag_seconds=50, fixed_lag_time=None):
    """
    Analyze the time lag between two signals using cross-correlation.
    Shows both separate vertical plots and combined plots.
    
    Parameters:
    merged_df: DataFrame containing the signals
    events_df: DataFrame containing event times with 'Start' and 'End' columns
    index: index of the event to analyze in events_df
    sec: seconds of padding before and after the event (default 50)
    signal1, signal2: names of the signals to compare
    sampling_rate: Hz (default 64 for your case)
    max_lag_seconds: maximum lag time to consider in seconds (default 50)
    fixed_lag_time: optional, if provided uses this lag time in seconds instead of calculating it
    
    Returns:
    lag_time: time lag in seconds
    lag_samples: lag in number of samples
    correlation: correlation coefficient
    """
    # Calculate start and end datetime
    start_datetime = events_df.iloc[index]['Start'] - pd.Timedelta(seconds=sec)
    end_datetime = events_df.iloc[index]['End'] + pd.Timedelta(seconds=sec)
    
    # Find subset of merged_df with index between start_datetime and end_datetime
    df = merged_df[(merged_df.index >= start_datetime) & (merged_df.index <= end_datetime)]
    
    # Get the signals
    s1 = df[signal1].values
    s2 = df[signal2].values
    
    # Compute cross-correlation
    correlation = signal.correlate(s1, s2, mode='full')
    lags = signal.correlation_lags(len(s1), len(s2), mode='full')
    
    # Process based on whether fixed lag is provided
    if fixed_lag_time is not None:
        # Convert fixed lag time to samples
        lag_samples = int(fixed_lag_time * sampling_rate)
        lag_time = fixed_lag_time
        
        # Find the corresponding correlation value
        # Find the index in lags array that is closest to lag_samples
        closest_lag_idx = np.argmin(np.abs(lags - lag_samples))
        max_correlation = correlation[closest_lag_idx]
        
        print(f"Using fixed lag time: {lag_time:.3f} seconds ({lag_samples} samples)")
    else:
        # Convert max_lag_seconds to samples
        max_lag_samples = int(max_lag_seconds * sampling_rate)
        
        # Create a mask for lags within our range
        valid_lags_mask = np.abs(lags) <= max_lag_samples
        
        # Apply mask to both lags and correlation
        valid_lags = lags[valid_lags_mask]
        valid_correlation = correlation[valid_lags_mask]
        
        # Find the lag with maximum correlation within the valid range
        max_corr_idx = np.argmax(valid_correlation)
        lag_samples = valid_lags[max_corr_idx]
        lag_time = lag_samples / sampling_rate
        max_correlation = valid_correlation[max_corr_idx]
        
        print(f"Calculated lag time: {lag_time:.3f} seconds ({lag_samples} samples)")
    
    # Create aligned version of second signal
    if lag_samples > 0:
        s2_aligned = np.pad(s2[:-lag_samples], (lag_samples, 0), mode='constant')
    else:
        s2_aligned = np.pad(s2[-lag_samples:], (0, -lag_samples), mode='constant')
    
    # FIGURE 1: Cross-correlation plot
    plt.figure(figsize=(15, 5))
    plt.plot(lags/sampling_rate, correlation)
    plt.axvline(x=lag_time, color='r', linestyle='--', label='Selected lag')
    
    if fixed_lag_time is None:
        # Add vertical lines showing the range we're considering
        plt.axvline(x=-max_lag_seconds, color='g', linestyle=':', label='Max lag limit')
        plt.axvline(x=max_lag_seconds, color='g', linestyle=':')
        title_suffix = f"(max lag: {max_lag_seconds} sec)"
    else:
        title_suffix = f"(fixed lag: {fixed_lag_time} sec)"
    
    plt.title(f'Cross-correlation between {signal1} and {signal2} {title_suffix}')
    plt.xlabel('Lag (seconds)')
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.legend()
    
    # FIGURE 2: Separate vertical plots
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot first original signal
    ax1.plot(df.index, s1, label=signal1)
    ax1.set_title(f'Original {signal1} Signal')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    # Add event lines
    ax1.axvline(x=events_df.iloc[index]['Start'], color='r', linestyle='--', linewidth=2, label='Event Start')
    ax1.axvline(x=events_df.iloc[index]['End'], color='g', linestyle='--', linewidth=2, label='Event End')
    ax1.legend(loc='upper right')
    
    # Plot second original signal
    ax2.plot(df.index, s2, label=signal2)
    ax2.set_title(f'Original {signal2} Signal')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    # Add event lines
    ax2.axvline(x=events_df.iloc[index]['Start'], color='r', linestyle='--', linewidth=2, label='Event Start')
    ax2.axvline(x=events_df.iloc[index]['End'], color='g', linestyle='--', linewidth=2, label='Event End')
    ax2.legend(loc='upper right')
    
    # Plot aligned second signal
    ax3.plot(df.index, s2_aligned, label=f'{signal2} (aligned)', color='green')
    ax3.set_title(f'Aligned {signal2} Signal (time lag = {lag_time:.3f} seconds, {lag_samples} samples)')
    ax3.legend(loc='upper right')
    ax3.grid(True)
    # Add event lines
    ax3.axvline(x=events_df.iloc[index]['Start'], color='r', linestyle='--', linewidth=2, label='Event Start')
    ax3.axvline(x=events_df.iloc[index]['End'], color='g', linestyle='--', linewidth=2, label='Event End')
    ax3.legend(loc='upper right')
    
    # Format x-axis to show time properly if index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # Add space between subplots
    
    # FIGURE 3: Combined plots showing both signals together
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot both original signals on the same subplot
    ax1.plot(df.index, s1, label=signal1, color='blue')
    ax1.plot(df.index, s2, label=signal2, color='orange')
    ax1.set_title(f'Original Signals: {signal1} and {signal2}')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    # Add event lines
    ax1.axvline(x=events_df.iloc[index]['Start'], color='r', linestyle='--', linewidth=2, label='Event Start')
    ax1.axvline(x=events_df.iloc[index]['End'], color='g', linestyle='--', linewidth=2, label='Event End')
    ax1.legend(loc='upper right')
    
    # Plot original signal 1 and aligned signal 2 on same subplot
    ax2.plot(df.index, s1, label=signal1, color='blue')
    ax2.plot(df.index, s2_aligned, label=f'{signal2} (aligned)', color='green')
    ax2.set_title(f'Comparison: {signal1} vs Aligned {signal2} (time lag = {lag_time:.3f} sec)')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    # Add event lines
    ax2.axvline(x=events_df.iloc[index]['Start'], color='r', linestyle='--', linewidth=2, label='Event Start')
    ax2.axvline(x=events_df.iloc[index]['End'], color='g', linestyle='--', linewidth=2, label='Event End')
    ax2.legend(loc='upper right')
    
    # Format x-axis to show time properly if index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # Add space between subplots
    
    return lag_time, lag_samples, max_correlation

# # Example usage
# fixed_lag = 32.375
# lag_time, lag_samples, correlation = analyze_time_lag(merged_df, obstructive_apnea_events, 
#                                                      index=100, sec=50)

# print(f"Time lag: {lag_time:.3f} seconds")
# print(f"Lag in samples: {lag_samples}")
# print(f"Maximum correlation coefficient: {correlation:.3f}")

def shift_breath_column(df, lag_time_seconds):
    """
    Shifts the 'breath' column ahead by lag_time_seconds and returns 
    a dataframe with only the overlapping rows.
    
    Parameters:
    df: DataFrame with datetime index
    lag_time_seconds: Time in seconds to shift the breath column
    
    Returns:
    DataFrame with breath column shifted and only overlapping rows
    """
    # Create a copy of the original dataframe
    result_df = df.copy()
    
    # Calculate the time delta for shifting
    time_delta = pd.Timedelta(seconds=lag_time_seconds)
    
    # Extract the breath column
    breath_series = df['breath'].copy()
    
    # Shift the index forward by the time delta (this shifts the data earlier)
    breath_series.index = breath_series.index - time_delta
    
    # Create a new dataframe with just the shifted breath column
    shifted_breath_df = pd.DataFrame({'breath_shifted': breath_series})
    
    # Merge the original dataframe with the shifted breath dataframe on the index
    merged_result = pd.merge(result_df, shifted_breath_df, 
                            left_index=True, right_index=True, 
                            how='inner')  # inner join keeps only overlapping rows
    
    # Replace the original breath column with the shifted one
    merged_result['breath'] = merged_result['breath_shifted']
    
    # Drop the temporary column
    merged_result = merged_result.drop('breath_shifted', axis=1)
    
    return merged_result

# # Example: Shift the breath column by 2.5 seconds
# lag_time = fixed_lag  # in seconds
# aligned_df = shift_breath_column(merged_df, lag_time)

# # Check the shapes to see how many rows were kept
# print(f"Original shape: {merged_df.shape}")
# print(f"Aligned shape: {aligned_df.shape}")