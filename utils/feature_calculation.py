# Current
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch, find_peaks, hilbert, butter, filtfilt
from datetime import timedelta
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def is_valid_data(values, min_length=2, min_std=1e-8):
    return len(values) >= min_length and np.std(values) >= min_std

def calculate_mean(values):
    return np.mean(values) if len(values) > 0 else np.nan

def calculate_std(values):
    return np.std(values) if len(values) > 0 else np.nan

def calculate_windowed_std(values, window_size=128):
    if len(values) < window_size:
        return np.std(values) if len(values) > 1 else np.nan  # Use global std if insufficient data
    
    windowed_std = [np.std(values[i:i+window_size]) for i in range(len(values) - window_size + 1)]
    return np.std(windowed_std)

def calculate_windowed_entropy(values, window_size=128):
    if len(values) < window_size:
        if len(values) > 1: # Use global std if insufficient data
            histogram, _ = np.histogram(np.std(values), bins='auto', density=True) 
            return entropy(histogram)
    else:
        windowed_std = [
            np.std(values[i:i+window_size])
            for i in range(len(values) - window_size + 1)
        ]
        histogram, _ = np.histogram(windowed_std, bins='auto', density=True)
        return entropy(histogram)

def calculate_min(values):
    return np.min(values) if len(values) > 0 else np.nan

def calculate_max(values):
    return np.max(values) if len(values) > 0 else np.nan

def calculate_median(values):
    return np.median(values) if len(values) > 0 else np.nan

def calculate_skewness(values):
    if not is_valid_data(values, min_length=3):
        return np.nan
    return skew(values)

def calculate_kurtosis(values):
    if not is_valid_data(values, min_length=4):
        return np.nan
    return kurtosis(values)

def calculate_rms(values):
    return np.sqrt(np.mean(np.square(values))) if len(values) > 0 else np.nan

def calculate_rms_filtered(values, fs = 64, freq_band = (0.1, 0.5)):
    """
    Calculate the RMS of a signal after filtering for a specific frequency band.

    Parameters:
    - values (array-like): Input signal values.
    - fs (float): Sampling frequency of the signal.
    - freq_band (tuple): Frequency band for filtering as (low_freq, high_freq).

    Returns:
    - float: RMS value of the filtered signal, or NaN if the input is invalid.
    """
    padlen = 27
    if len(values) == 0 or len(values) <= padlen or fs <= 0 or not (isinstance(freq_band, tuple) and len(freq_band) == 2):
        return np.nan

    # Design a bandpass filter
    low, high = freq_band
    nyquist = 0.5 * fs
    low = low / nyquist
    high = high / nyquist

    # Check frequency band validity
    if low <= 0 or high >= 1 or low >= high:
        return np.nan

    # Butterworth filter design
    b, a = butter(N=4, Wn=[low, high], btype='band')

    # Apply the filter
    filtered_values = filtfilt(b, a, values)

    # Calculate RMS of the filtered signal
    rms_value = np.sqrt(np.mean(np.square(filtered_values)))
    return rms_value

def calculate_iqr(values):
    if len(values) > 0:
        return np.percentile(values, 75) - np.percentile(values, 25)
    return np.nan

def calculate_line_length(values):
    return 10 * 64 * np.sum(np.abs(np.diff(values))) / (len(values) - 1) if len(values) > 1 else np.nan

def calculate_variance_of_amplitude(values):
    if not is_valid_data(values, min_length=2):
        return np.nan
    return np.var(values)

def calculate_slope_of_amplitude_changes(values):
    diff_values = np.diff(values)
    return np.mean(diff_values) if len(diff_values) > 0 else np.nan

def calculate_amplitude_envelope(values):
    if len(values) == 0:
        return np.nan
    analytic_signal = hilbert(values)
    amplitude_envelope = np.abs(analytic_signal)
    return np.mean(amplitude_envelope)

def calculate_zero_crossing_rate(values):
    return 64 * 10 * ((values[:-1] * values[1:]) < 0).sum() / len(values) if len(values) > 1 else np.nan

def calculate_threshold_zero_crossing_rate(values, threshold_fraction=0.1):
    if len(values) <= 1:
        return np.nan
    max_amplitude = np.max(np.abs(values))
    threshold = max_amplitude * threshold_fraction
    crossing_count = ((values[:-1] < -threshold) & (values[1:] >= -threshold)) | \
                     ((values[:-1] > threshold) & (values[1:] <= threshold))
    return 64 * 10 * np.sum(crossing_count) / len(values)

def calculate_threshold_zero_crossing_rate_adaptive(values, window_size=100, threshold_fraction=0.1):
    if len(values) <= 1 or len(values) < window_size:
        return np.nan
    zcr = []
    for i in range(0, len(values) - window_size + 1, window_size):
        window = values[i:i+window_size]
        threshold = np.max(np.abs(window)) * threshold_fraction
        crossings = ((window[:-1] < -threshold) & (window[1:] >= -threshold)) | \
                    ((window[:-1] > threshold) & (window[1:] <= threshold))
        zcr.append(np.sum(crossings) / window_size)
    return np.mean(zcr)

def calculate_autocorrelation(values):
    values = np.array(values)
    if len(values) < 2 or np.std(values) < 1e-8:
        return np.nan
    autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
    return autocorr if not np.isnan(autocorr) else np.nan

def calculate_autocorrelation_multi_lag(values, max_lag=320):
    if len(values) <= max_lag:
        return np.nan
    autocorrs = [np.corrcoef(values[:-lag], values[lag:])[0, 1] for lag in range(192, max_lag+1)]
    autocorrs = [ac for ac in autocorrs if not np.isnan(ac)]
    return np.mean(autocorrs) if autocorrs else np.nan

# def find_pos_neg_peaks(values, prominence=0.3, distance=128, height=0.8, width=50):
# def find_pos_neg_peaks(values, prominence=0.1, distance=64, height=0.3, width=50):
def find_pos_neg_peaks(values, prominence=0, distance=1, height=0, width=0):
    peaks_pos, properties_pos = find_peaks(values, prominence=prominence, distance=distance, height=height, width=width)
    peaks_neg, properties_neg = find_peaks(-values, prominence=prominence, distance=distance, height=height, width=width)
    peaks = np.concatenate((peaks_pos, peaks_neg), axis=0)
    properties = {**properties_pos, **properties_neg}
    return peaks, properties

def calculate_peak_to_peak(values):
    values = np.array(values)
    return np.ptp(values) if len(values) > 0 else np.nan

def calculate_number_of_peaks(values):
    # Find peaks with specified prominence and minimum distance between peaks
    peaks, properties = find_pos_neg_peaks(values)

    # Calculate the number of peaks normalized by the signal length
    return 10 * 64 * len(peaks) / (len(values) - 1)

def calculate_peak_prominence(values):
    # peaks, properties = find_peaks(values, prominence=1)
    peaks, properties = find_pos_neg_peaks(values)

    return np.mean(properties['prominences']) if len(peaks) > 0 else np.nan

def calculate_peak_width(values):
    peaks, properties = find_pos_neg_peaks(values)

    return np.mean(properties['widths']) if 'widths' in properties and len(peaks) > 0 else np.nan

def calculate_peak_to_peak_variability(values):
    peaks, properties = find_pos_neg_peaks(values)
    
    # Calculate distances between consecutive peaks
    if len(peaks) < 2:
        return np.nan  # Not enough peaks to calculate variability
    
    peak_to_peak_distances = np.diff(peaks)
    
    # Calculate variability (standard deviation of distances)
    return np.std(peak_to_peak_distances)

def calculate_power_spectral_density(values, sampling_rate=64.0, window='hamming'):
    if len(values) < 1:
        return np.nan
    nperseg = min(256, len(values))
    if nperseg < 1:
        return np.nan
    f, Pxx = welch(values, fs=sampling_rate, nperseg=nperseg, window=window)
    return np.sum(Pxx)

def calculate_band_power(values, sampling_rate=64.0, bands=[(0.1, 0.5), (0.5, 1)], window='hamming'):
    if len(values) < 1:
        # return [np.nan] * len(bands)
        return np.nan
    f, Pxx = welch(values, fs=sampling_rate, nperseg=min(256, len(values)), window=window)
    band_powers = []
    for (low, high) in bands:
        idx_band = np.logical_and(f >= low, f <= high)
        band_power = np.trapz(Pxx[idx_band], f[idx_band])
        band_powers.append(band_power)
    return band_powers[0]

def calculate_spectral_entropy(values, sampling_rate=64.0, window='hamming'):
    if len(values) < 1:
        return np.nan
    nperseg = min(256, len(values))
    if nperseg < 1:
        return np.nan
    f, Pxx = welch(values, fs=sampling_rate, nperseg=nperseg, window=window)
    
    # Total power
    total_power = np.sum(Pxx)
    # Spectral Entropy
    normalized_psd = Pxx / total_power if total_power > 0 else np.zeros_like(Pxx)
    spectral_entropy = entropy(normalized_psd, base=2)
    return spectral_entropy

def calculate_shannon_entropy(values):
    if len(values) == 0:
        return np.nan
    hist, _ = np.histogram(values, bins='fd', density=True)  # 'fd' is the Freedman-Diaconis rule
    hist = hist[hist > 0]
    hist_sum = hist.sum()
    if hist_sum == 0:
        return np.nan
    return -np.sum(hist * np.log(hist)) / hist_sum

def calculate_dominant_frequency(values, sampling_rate=64.0, window='hamming'):
    if len(values) == 0:
        return np.nan
    
    nperseg = min(256, len(values))
    try:
        f, Pxx = welch(values, fs=sampling_rate, nperseg=nperseg, window=window)
        if len(Pxx) == 0:
            return np.nan
        dominant_freq = f[np.argmax(Pxx)]
        return dominant_freq
    except Exception as e:
        # Log the exception if needed
        return np.nan

def calculate_wavelet_coefficients(values, wavelet='harr', level=3):
    if len(values) == 0:
        return np.nan
    # Dynamic level adjustment
    max_level = int(np.floor(np.log2(len(values))))
    level = min(max_level, level)
    if level < 1:
        return np.nan  # Insufficient length for even one level
    try:
        coeffs = pywt.wavedec(values, wavelet, level=level)
        # Example: Return the mean of the approximation coefficients at the highest level
        return np.mean(coeffs[0]) if len(coeffs[0]) > 0 else np.nan
    except Exception as e:
        # Log the exception if needed
        return np.nan
    
def calculate_wavelet_sum(values, wavelet='harr', level=5):
    if len(values) == 0:
        return np.nan
    # Dynamic level adjustment
    max_level = int(np.floor(np.log2(len(values))))
    level = min(max_level, level)
    if level < 1:
        return np.nan  # Insufficient length for even one level
    try:
        coeffs = pywt.wavedec(values, wavelet, level=level)
        # Example: Return the mean of the approximation coefficients at the highest level
        return sum(np.sum(np.abs(c)) / len(c) for c in coeffs if len(c) > 0)
    except Exception as e:
        # Log the exception if needed
        return np.nan


FEATURE_FUNCTIONS = {
    'Mean': calculate_mean,
    'Std': calculate_std,
    'WindowStd': calculate_windowed_std,
    'WindowEntropy': calculate_windowed_entropy,
    'Min': calculate_min,
    'Max': calculate_max,
    'Median': calculate_median,
    'Skewness': calculate_skewness,
    'Kurtosis': calculate_kurtosis,
    'RMS': calculate_rms,
    'RMS_filtered': calculate_rms_filtered,
    'IQR': calculate_iqr,
    'LineLength': calculate_line_length,
    'Variance': calculate_variance_of_amplitude,
    'Slope': calculate_slope_of_amplitude_changes,
    'ZeroCrossingRate': calculate_zero_crossing_rate,
    'ThresholdZCR': calculate_threshold_zero_crossing_rate,
    'Autocorrelation': calculate_autocorrelation,
    'AutocorrelationLagged': calculate_autocorrelation_multi_lag,
    'PeakToPeak': calculate_peak_to_peak,
    'NumPeaks': calculate_number_of_peaks,
    'PeakProminence': calculate_peak_prominence,
    'PeakWidth': calculate_peak_width,
    'PeakVariance': calculate_peak_to_peak_variability,
    'PowerSpectralDensity': calculate_power_spectral_density,
    'BandPower': calculate_band_power,
    'SpectralEntropy': calculate_spectral_entropy,
    'ShannonEntropy': calculate_shannon_entropy,
    'DominantFrequency': calculate_dominant_frequency,
    'WaveletCoeff': calculate_wavelet_coefficients,
    'WaveletSum': calculate_wavelet_sum,
    'AmplitudeEnvelope': calculate_amplitude_envelope
    # Add more features as needed
}

def batch_compute_features(
    segmented_df, 
    signal_list, 
    feature_dict, 
    label_col='label',    # or 'Label' if that's what it's called in your DataFrame
    n_jobs=1
):
    """
    Batch compute the features for each signal in each row of segmented_df using
    the provided feature_dict and return a new DataFrame.

    Parameters
    ----------
    segmented_df : pd.DataFrame
        The DataFrame produced by your segmentation function. It contains:
            - multiple signal columns (each as a 1D np.array)
            - a label column (e.g. 'label')
    signal_list : list of str
        The list of signal columns in segmented_df for which you want to compute features.
    feature_dict : dict
        A dictionary mapping feature names to feature functions:
            {
                'Mean': calculate_mean,
                'Std': calculate_std,
                ...
            }
    label_col : str, optional
        Name of the column in segmented_df that contains your label (default='label').
    n_jobs : int, optional
        Number of CPU cores to use for parallel processing (default=1). Use -1 for all cores.

    Returns
    -------
    features_df : pd.DataFrame
        A new DataFrame where each row corresponds to one record (segment), and
        each column corresponds to a feature of a particular signal.
        The final column is the label taken from segmented_df.
    """
    def compute_features_for_row(row):
        """
        Compute all desired features for a single row of segmented_df.
        Return a dict where keys = 'signal_featureName', value = feature_value.
        """
        row_features = {}

        # For each signal in signal_list, compute all features in feature_dict
        for signal_col in signal_list:
            values = row[signal_col]  # This is a 1D np.array
            for feat_name, feat_func in feature_dict.items():
                col_name = f"{signal_col}_{feat_name}"
                row_features[col_name] = feat_func(values)

        # Attach the label as the last column (we'll rename it to "Label" if desired)
        row_features["Label"] = row[label_col]

        return row_features

    # Parallel (or serial) execution over the rows of segmented_df
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_features_for_row)(row)
        for _, row in segmented_df.iterrows()
    )

    # Convert the list of dicts into a DataFrame
    features_df = pd.DataFrame(results)

    return features_df


def process_single_event(event_row, df_subset, signal_list, sampling_rate, padding_seconds = 30, extra_padding = 5):
    start_time = event_row['Start'] - timedelta(seconds=extra_padding)
    end_time = event_row['End'] + timedelta(seconds=extra_padding)
    padded_start_time = start_time - timedelta(seconds=padding_seconds)
    padded_end_time = end_time + timedelta(seconds=padding_seconds)

    relevant_data = df_subset[(df_subset.index >= padded_start_time) & (df_subset.index <= padded_end_time)]
    exact_data = df_subset[(df_subset.index >= start_time) & (df_subset.index <= end_time)]
    head_data = df_subset[(df_subset.index >= padded_start_time) & (df_subset.index < start_time)]
    tail_data = df_subset[(df_subset.index > end_time) & (df_subset.index <= padded_end_time)]
    head_tail_data = pd.concat([head_data, tail_data])

    occurrence_entry = {
        'Start': start_time,
        'End': end_time,
        'Padded_Start': padded_start_time,
        'Padded_End': padded_end_time,
        'Time': relevant_data.index
    }

    for signal in signal_list:
        occurrence_entry[signal] = relevant_data[signal].values
        occurrence_entry[f'Exact_{signal}'] = exact_data[signal].values
        occurrence_entry[f'Head_Tail_{signal}'] = head_tail_data[signal].values

    stats_entry = {}
    for signal in signal_list:
        exact_values = occurrence_entry.get(f'Exact_{signal}', [])
        head_tail_values = occurrence_entry.get(f'Head_Tail_{signal}', [])

        stats_entry[signal] = {
            'Exact': {feature: func(exact_values) for feature, func in FEATURE_FUNCTIONS.items()},
            'Head_Tail': {feature: func(head_tail_values) for feature, func in FEATURE_FUNCTIONS.items()}
        }

    return occurrence_entry, stats_entry

def prepare_structured_data_parallel(merged_df, events_df, signal_list, sampling_rate=1.0, padding_seconds=30, extra_padding = 5, n_jobs=-1):
    structured_data = {}
    df_subset = merged_df[signal_list]

    for event_type in events_df['Name'].unique():
        event_occurrences = events_df[events_df['Name'] == event_type]

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_event)(
                event_row, df_subset, signal_list, sampling_rate, padding_seconds, extra_padding
            ) for _, event_row in event_occurrences.iterrows()
        )

        occurrences_list, stats_list = zip(*results) if results else ([], [])

        structured_data[event_type] = {
            'Occurrences': list(occurrences_list),
            'Stats': list(stats_list)
        }

    return structured_data

def analyze_and_plot_event_data(structured_data, event_type, signal_list, max_plots=5):
    if event_type not in structured_data:
        print(f"Event type '{event_type}' not found in structured_data.")
        return

    event_data = structured_data[event_type]['Occurrences']
    stats = structured_data[event_type]['Stats']

    for i, (occurrence, stat_entry) in enumerate(zip(event_data, stats)):
        if i >= max_plots:
            print(f"Reached maximum number of plots ({max_plots}).")
            break

        start_time = occurrence['Start']
        end_time = occurrence['End']
        signals = signal_list
        num_signals = len(signals)
        fig, axes = plt.subplots(num_signals, 2, figsize=(18, 5 * num_signals))
        if num_signals == 1:
            axes = np.array([axes])  # Ensure axes is 2D

        for j, signal in enumerate(signals):
            ax_signal, ax_box = axes[j]
            signal_data = occurrence.get(signal, [])
            time_data = occurrence['Time']
            exact_values = occurrence.get(f'Exact_{signal}', [])
            head_tail_values = occurrence.get(f'Head_Tail_{signal}', [])

            if len(time_data) != len(signal_data):
                continue

            # Plot Signal
            ax_signal.plot(time_data, signal_data, label=signal)
            ax_signal.axvline(x=start_time, color='r', linestyle='--', linewidth=2, label='Start')
            ax_signal.axvline(x=end_time, color='r', linestyle='--', linewidth=2, label='End')
            ax_signal.set_title(f'{event_type} - {signal} (# {i + 1})')
            ax_signal.set_xlabel('Time')
            ax_signal.set_ylabel(f'{signal} Value')
            ax_signal.legend()

            # Feature Stats
            exact_stats = stat_entry[signal]['Exact']
            exact_textstr = '\n'.join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                                       for k, v in exact_stats.items()])
            exact_props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
            ax_signal.text(0.05, 0.95, exact_textstr, transform=ax_signal.transAxes, fontsize=9,
                           verticalalignment='top', bbox=exact_props)

            head_tail_stats = stat_entry[signal]['Head_Tail']
            head_tail_textstr = '\n'.join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                                          for k, v in head_tail_stats.items()])
            head_tail_props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
            ax_signal.text(0.95, 0.95, head_tail_textstr, transform=ax_signal.transAxes, fontsize=9,
                           verticalalignment='top', horizontalalignment='right', bbox=head_tail_props)

            # Box Plot
            if len(exact_values) > 0 and len(head_tail_values) > 0:
                ax_box.boxplot([exact_values, head_tail_values], labels=['Exact', 'Head_Tail'])
                ax_box.set_title(f'{event_type} - {signal} Box Plot (# {i + 1})')
                ax_box.set_ylabel(f'{signal} Value')
            else:
                ax_box.text(0.5, 0.5, 'Not enough data for box plot',
                            horizontalalignment='center', verticalalignment='center')
                ax_box.set_title(f'{event_type} - {signal} Box Plot (# {i + 1})')
                ax_box.set_ylabel(f'{signal} Value')

        plt.tight_layout()
        plt.show()
        
def create_feature_dataframe(structured_data, event_type, signal_list):
    """
    Create a DataFrame from structured data for a specific event type.

    Each event occurrence will generate two rows:
    1. One for the 'Exact' segment with label True.
    2. One for the 'Head_Tail' segment with label False.

    Parameters:
    - structured_data (dict): The structured data containing feature statistics.
    - event_type (str): The event type to process (e.g., 'Central Apnea').
    - signal_list (list): List of signal names used in feature extraction.

    Returns:
    - features_df (pd.DataFrame): A DataFrame with features and labels.
    """

    # Check if the event_type exists in structured_data
    if event_type not in structured_data:
        raise ValueError(f"Event type '{event_type}' not found in structured_data.")

    # Initialize a list to collect all rows
    rows = []

    # Iterate over each stats_entry corresponding to the event_type
    for stats_entry in structured_data[event_type]['Stats']:
        # Initialize dictionaries for 'Exact' and 'Head_Tail'
        exact_row = {}
        head_tail_row = {}

        # Iterate through each signal in signal_list
        for signal in signal_list:
            # Prefix features with signal name for clarity
            for feature, value in stats_entry[signal]['Exact'].items():
                column_name = f"{signal}_{feature}"
                exact_row[column_name] = value

            for feature, value in stats_entry[signal]['Head_Tail'].items():
                column_name = f"{signal}_{feature}"
                head_tail_row[column_name] = value

        # Assign labels
        exact_row['Label'] = True
        head_tail_row['Label'] = False

        # Append the rows to the list
        rows.append(exact_row)
        rows.append(head_tail_row)

    # Create the DataFrame
    features_df = pd.DataFrame(rows)

    # Generate a list of all possible feature columns based on FEATURE_FUNCTIONS
    all_feature_columns = []
    for signal in signal_list:
        for feature in FEATURE_FUNCTIONS.keys():
            column_name = f"{signal}_{feature}"
            all_feature_columns.append(column_name)

    # Ensure all expected columns are present in the DataFrame
    for col in all_feature_columns:
        if col not in features_df.columns:
            features_df[col] = np.nan  # Assign NaN if the feature is missing

    # Reorder columns: Features first, then Label
    features_df = features_df[all_feature_columns + ['Label']]

    return features_df