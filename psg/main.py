# main.py

from xml_data_extraction import XMLProcessor
from psg_data_extraction import PSGDataProcessor
from datetime import datetime

def main():
    # Example usage of XMLProcessor
    xml_file_path = 'path_to_your_xml_file.xml'
    xml_start_datetime_str = "2024-06-20 22:02:34"
    
    xml_processor = XMLProcessor(xml_file_path, xml_start_datetime_str)
    xml_processor.load_and_pretty_print_xml()
    xml_processor.extract_scored_events()
    xml_processor.extract_sleep_stages()
    xml_processor.plot_sleep_stages()
    xml_processor.plot_sleep_stages_by_code()
    xml_processor.analyze_sleep_stages()
    
    # Example usage of PSGDataProcessor
    psg_file_path = 'path_to_your_edf_file.edf'
    
    psg_processor = PSGDataProcessor()
    psg_processor.load_data(psg_file_path)
    psg_processor.psg_plot()
    psg_processor.print_file_info(psg_file_path, 'file')
    psg_processor.print_file_info(psg_file_path, 'label')
    psg_processor.print_file_info(psg_file_path, 'signal')
    
    # Plot signals over time
    start_datetime = datetime(2024, 3, 7, 22, 10, 0)
    end_datetime = datetime(2024, 3, 7, 22, 11, 0)
    all_types = list(extracted_data.keys())
    data_types = ['ECG', 'Thor']
    
    print(f"Start Timestamp: {start_datetime}, End Timestamp: {end_datetime}")
    extracted_data = psg_processor.extract_segment_by_timestamp(start_datetime, end_datetime, data_types)
    psg_processor.plot_data(extracted_data['ECG'], 'ECG', psg_processor.sampling_rate)
    
    # Plot comparison between signals
    psg_processor.compare_plot(extracted_data, data_types, psg_processor.sampling_rate)
    
    # Plot ECG signal
    ecg_signals, ecg_info = psg_processor.ecg_diagram(extracted_data['ECG'])
    
    # Plot RSP signal
    rsp_signals, rsp_info = psg_processor.rsp_diagram(extracted_data['Thor'])
    
    # Plot multiple PSG signals
    multi_data_types = ['ECG', 'Thor', 'EMG_L', 'E1-M2']
    multi_extracted_data = psg_processor.extract_segment_by_timestamp(start_datetime, end_datetime, multi_data_types)
    psg_processor.signals_diagram(multi_extracted_data)

if __name__ == "__main__":
    main()
