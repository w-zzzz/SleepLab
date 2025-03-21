def plot_fft(signal, sample_rate, title):
    """
    Plots the FFT of a given signal and returns the frequency and amplitude of the highest point.

    Parameters:
    signal (np.array): The input signal.
    sample_rate (float): The sample rate of the signal.
    title (str): The title of the plot.

    Returns:
    tuple: The frequency and amplitude of the highest point in the FFT plot.
    """
    # Remove the DC component
    signal = signal - np.mean(signal)
    
    # Compute the FFT
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=1/sample_rate)

    # Find the highest point in the FFT plot
    positive_freqs = fft_freq[:len(fft_freq)//2]
    positive_amplitudes = np.abs(fft_result[:len(fft_result)//2])
    max_index = np.argmax(positive_amplitudes)
    max_freq = positive_freqs[max_index]
    max_amplitude = positive_amplitudes[max_index]

    # Plot the FFT result
    plt.figure(figsize=(12, 6))
    plt.plot(positive_freqs, positive_amplitudes)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0, 1)  # Limit the plot to the 0 to 5 Hz range
    plt.grid(True)
    plt.show()

    return max_freq, max_amplitude

def visualize_events_in_window(xml_processor, merged_df, target_event_type=None, idx=None, 
                             padding_seconds=None, start_time=None, end_time=None,
                             event_list=None):
    """
    Visualizes events within a specified time window with flexible behavior depending on parameters.
    
    Behavior modes:
    1. No parameters: Visualizes all events across the entire merged_df timespan
    2. Only start_time/end_time: Visualizes all events in the specified window
    3. target_event_type only: Highlights all events of specified type
    4. idx, target_event_type, padding_seconds: Focuses on a specific event (original behavior)
    
    Parameters:
    -----------
    xml_processor : XMLProcessor
        The XMLProcessor object containing the events data
    merged_df : pandas.DataFrame
        DataFrame with time-indexed signals to plot
    target_event_type : str, optional
        The type of event to highlight (e.g., 'Obstructive Apnea')
    idx : int, optional
        Index of the specific event in the filtered events list
    padding_seconds : int, optional
        Number of seconds to pad before and after the main event
    start_time : datetime, optional
        Start time of the window to visualize
    end_time : datetime, optional
        End time of the window to visualize
    event_list : list, optional
        List of event names to filter for (only these event types will be shown)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    # Apply event type filtering if event_list is provided
    if event_list is not None:
        all_events = xml_processor.events.copy()
        filtered_events = all_events[all_events['Name'].isin(event_list)]
        
        if filtered_events.empty:
            print(f"Warning: No events found with names: {event_list}")
            print(f"Available event types: {all_events['Name'].unique()}")
            # Continue with all events if filter would result in empty set
            working_events = all_events
        else:
            working_events = filtered_events
            print(f"Filtered to show only these event types: {event_list}")
    else:
        working_events = xml_processor.events
    
    # Determine the time window based on inputs
    if idx is not None and target_event_type is not None and padding_seconds is not None:
        # Scenario 4: Specific event with padding (original behavior)
        target_events = working_events[working_events['Name'] == target_event_type]
        
        if target_events.empty:
            raise ValueError(f"No events of type '{target_event_type}' found after filtering")
        
        if idx >= len(target_events):
            raise ValueError(f"Index {idx} is out of bounds for events of type '{target_event_type}' (max index: {len(target_events)-1})")
            
        # Get the target event
        target_event = target_events.iloc[idx]
        target_idx = target_events.index[idx]  # Store the actual index for reference
        
        # Define the padded window
        start_datetime = target_event['Start'] - pd.Timedelta(seconds=padding_seconds)
        end_datetime = target_event['End'] + pd.Timedelta(seconds=padding_seconds)
        
        title = f'Events Window: Target {target_event_type} (Index: {target_idx})'
        
    elif start_time is not None and end_time is not None:
        # Scenario 2: Custom time window
        start_datetime = start_time
        end_datetime = end_time
        target_idx = None
        
        title = f'Events between {start_datetime.strftime("%Y-%m-%d %H:%M:%S")} and {end_datetime.strftime("%Y-%m-%d %H:%M:%S")}'
        
    else:
        # Scenario 1: Full dataset range
        start_datetime = merged_df.index.min()
        end_datetime = merged_df.index.max()
        target_idx = None
        
        title = 'All Events in Dataset'
    
    # Find subset of merged_df within the time window
    window_df = merged_df[(merged_df.index >= start_datetime) & (merged_df.index <= end_datetime)]
    
    if window_df.empty:
        raise ValueError(f"No data found in the specified time range: {start_datetime} to {end_datetime}")
    
    # Find ALL events (of filtered types) within the time window
    all_window_events = working_events[
        (working_events['Start'] >= start_datetime) & 
        (working_events['End'] <= end_datetime)
    ].copy()
    
    if all_window_events.empty:
        print(f"No events found in the time window {start_datetime} to {end_datetime}")
        # Plot the signals anyway, but without events
        fig, ax = plt.subplots(figsize=(20, 4))
        for column in ['breath', 'Thor']:
            if column in window_df.columns:
                ax.plot(window_df.index, window_df[column], label=column)
        ax.set_xlabel('Time')
        ax.set_ylabel('Signal Amplitude')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        return fig
    
    # Add a column to identify the target event (if any)
    if target_idx is not None:
        all_window_events['is_target'] = all_window_events.index == target_idx
    else:
        all_window_events['is_target'] = False
    
    # Add a column to identify target event type (for Scenario 3)
    if target_event_type is not None:
        all_window_events['is_target_type'] = all_window_events['Name'] == target_event_type
    else:
        all_window_events['is_target_type'] = False
    
    # Sort events by start time
    all_window_events = all_window_events.sort_values('Start')
    
    # Get unique event types in the window
    event_types = all_window_events['Name'].unique()
    
    # Create a color map for different event types
    import matplotlib.colors as mcolors
    
    # Choose colors for event types
    other_colors = list(mcolors.TABLEAU_COLORS.values())
    color_map = {event_type: other_colors[i % len(other_colors)] 
                for i, event_type in enumerate(event_types)}
    
    # Print information about events in chronological order
    print(f"\n{'='*100}")
    
    if target_idx is not None:
        print(f"All events within the padded window ({padding_seconds}s before and after)")
        print(f"Target event: {target_event_type} (Index: {target_idx})")
    else:
        print(f"All events within the time window: {start_datetime} to {end_datetime}")
    
    if event_list is not None:
        print(f"Filtered to only include these event types: {event_list}")
    
    print(f"{'='*100}")
    print(f"{'Index':<10} {'Event Type':<25} {'Start Time':<25} {'End Time':<25} {'Duration (s)':<15} {'Is Target':<10}")
    print(f"{'-'*100}")
    
    for i, event in all_window_events.iterrows():
        duration = (event['End'] - event['Start']).total_seconds()
        target_marker = "* TARGET *" if event['is_target'] else ""
        print(f"{i:<10} {event['Name']:<25} {event['Start']!s:<25} {event['End']!s:<25} {duration:<15.2f} {target_marker:<10}")
    
    print(f"{'='*100}")
    print(f"Total events in window: {len(all_window_events)}")
    print(f"Event types present: {', '.join(event_types)}")
    print(f"{'='*100}\n")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(20, 4))
    
    # Plot the signals
    for column in ['breath', 'Thor']:
        if column in window_df.columns:
            ax.plot(window_df.index, window_df[column], label=column)
    
    # Create legend handles for event types
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # Draw events on the plot
    for event_type in event_types:
        events_of_this_type = all_window_events[all_window_events['Name'] == event_type]
        
        # Get color for this event type
        color = color_map[event_type]
        
        # Use same line style and width for all event types
        linewidth = 1.5
        linestyle = '--'
        alpha_value = 0.5
        
        # Add to legend once per event type
        legend_elements.append(Line2D([0], [0], color=color, lw=linewidth, 
                                    linestyle=linestyle, 
                                    label=f"{event_type} ({len(events_of_this_type)})"))
        
        # Draw each event of this type
        for i, event in events_of_this_type.iterrows():
            is_target_instance = event['is_target']
            is_target_type = event['is_target_type']
            
            # All events have the same style
            line_width = linewidth
            span_alpha = alpha_value
            
            # Draw vertical lines at start and end
            ax.axvline(x=event['Start'], color=color, linestyle=linestyle, linewidth=line_width)
            ax.axvline(x=event['End'], color=color, linestyle=linestyle, linewidth=line_width)
            
            # Add spans
            ax.axvspan(event['Start'], event['End'], alpha=span_alpha, color=color)
            
            # Add event index text with appropriate styling
            mid_time = event['Start'] + (event['End'] - event['Start'])/2
            y_pos = 0.1 if event_type == target_event_type else 0.05
            
            # Style based on if it's the target instance or target type
            if is_target_instance:
                # Target instance (specific event)
                ax.text(mid_time, y_pos, f"{i}", transform=ax.get_xaxis_transform(), 
                       ha='center', va='bottom', fontsize=10, color='red', weight='bold',
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
            elif is_target_type and target_idx is None:
                # For scenario 3: Highlight all events of target type
                ax.text(mid_time, y_pos, f"{i}", transform=ax.get_xaxis_transform(), 
                       ha='center', va='bottom', fontsize=9, color='red', weight='bold',
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
            else:
                # Normal events
                ax.text(mid_time, y_pos, f"{i}", transform=ax.get_xaxis_transform(), 
                       ha='center', va='bottom', fontsize=8, 
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Set plot labels and properties
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal Amplitude')
    ax.set_title(title + (" (filtered)" if event_list is not None else ""))
    
    # Add signal traces to legend
    for column in ['breath', 'Thor']:
        if column in window_df.columns:
            legend_elements.append(Line2D([0], [0], color=ax.get_lines()[-2 if 'breath' in window_df.columns and 'Thor' in window_df.columns else -1].get_color(), 
                                         lw=2, label=column))
    
    # Place legend outside the plot
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), 
              borderaxespad=0, fontsize=10)
    
    ax.grid(True)
    
    # Format time axis
    plt.gcf().autofmt_xdate()
    
    # Add information about the time window
    time_window = (end_datetime - start_datetime).total_seconds()
    plt.figtext(0.3, 0.98, f"Time Window: {time_window:.0f} seconds", 
                ha="center", fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # If we have a target event, add info about its duration
    if target_idx is not None:
        event_duration = (target_event['End'] - target_event['Start']).total_seconds()
        plt.figtext(0.15, 0.98, f"Target Duration: {event_duration:.2f}s", 
                    ha="center", fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add more spacing at the top for the annotations
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    return fig

