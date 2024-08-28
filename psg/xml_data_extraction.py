import os
import pandas as pd
import matplotlib.pyplot as plt
from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from collections import Counter
# from .constants import XML_PATH, START_DATETIME_STR

plt.rcParams['figure.figsize'] = (18, 3)

class XMLProcessor:
    def __init__(self, file_path, start_datetime):
        """
        Initialize the XMLProcessor with file path and start datetime.

        Args:
        file_path (str): Path to the XML file.
        start_datetime: Start datetime in the format 'YYYY-MM-DD HH:MM:SS'.
        """
        self.file_path = file_path
        self.start_datetime = start_datetime
        self.root = None
        self.events = None
        self.sleep_stages = None
        self.stage_mapping = {
            5: 'REM Sleep',
            0: 'Wakefulness (W)',
            1: 'NREM Sleep Stage 1 (N1)',
            2: 'NREM Sleep Stage 2 (N2)',
            3: 'NREM Sleep Stage 3 (N3)',
            9: 'Movement Time (MT)'
        }
        self.epoch_length = None

    def load(self):
        """
        Load and pretty print the XML content.
        Saves the pretty-printed XML to a new file.
        """
        try:
            with open(self.file_path, 'r') as file:
                xml_content = file.read()
            self.root = ET.fromstring(xml_content)
            pretty_xml = parseString(xml_content).toprettyxml()

            # Prefix 'pretty_' to the filename
            dir_path, filename = os.path.split(self.file_path)
            modified_filename = 'pretty_' + filename
            save_path = os.path.join(dir_path, modified_filename)
            
            with open(save_path, 'w') as file:
                file.write(pretty_xml)
            
            # print(pretty_xml)
            self.extract_scored_events()
            self.extract_sleep_stages()
            print("Load successful!")
            print("Saved the pretty-printed XML to path: ", save_path)
        except Exception as e:
            print(f"Error loading or pretty printing XML: {e}")
    
    def extract_scored_events(self):
        """
        Extract scored events from the XML and create a DataFrame.
        """
        try:
            scored_events = []
            for event in self.root.findall(".//ScoredEvent"):
                event_data = {
                    'Name': event.find('Name').text,
                    'Start': float(event.find('Start').text),
                    'Duration': float(event.find('Duration').text) if event.find('Duration') is not None else 0,
                    'Input': event.find('Input').text if event.find('Input') is not None else ''
                }
                scored_events.append(event_data)
            
            self.events = pd.DataFrame(scored_events)
            self.events['Actual Start (sec)'] = self.events['Start'].round(2)
            self.events['Duration'] = self.events['Duration'].round(2)
            self.events['Start'] = self.events['Actual Start (sec)'].apply(lambda x: self.start_datetime + timedelta(seconds=x))
            self.events['End'] = self.events.apply(lambda row: row['Start'] + timedelta(seconds=row['Duration']), axis=1)
            self.events = self.events[['Name', 'Actual Start (sec)', 'Duration', 'Start', 'End', 'Input']]
            print(self.events)
        except Exception as e:
            print(f"Error extracting scored events: {e}")
    
    def extract_sleep_stages(self):
        """
        Extract sleep stages from the XML and create a DataFrame.
        """
        try:
            sleep_stages = [int(stage.text) for stage in self.root.find('SleepStages')]
            self.epoch_length = int(self.root.find('EpochLength').text.strip())
            time_axis = [i * self.epoch_length for i in range(len(sleep_stages))]
            actual_start_times = [self.start_datetime + timedelta(seconds=t) for t in time_axis]

            mapped_stages = [self.stage_mapping[stage] for stage in sleep_stages]

            self.sleep_stages = pd.DataFrame({
                'Time (seconds)': time_axis,
                'Start Time': actual_start_times,
                'Sleep Stage Code': sleep_stages,
                'Sleep Stage': mapped_stages
            })
            print(self.sleep_stages)
        except Exception as e:
            print(f"Error extracting sleep stages: {e}")

    def plot_sleep_stages(self):
        """
        Plot sleep stages over time.
        """
        try:
            colors = plt.cm.get_cmap('tab10', len(self.stage_mapping))
            added_labels = set()

            for i in range(len(self.sleep_stages) - 1):
                stage_label = self.stage_mapping[self.sleep_stages['Sleep Stage Code'][i]]
                if stage_label not in added_labels:
                    plt.plot(self.sleep_stages['Time (seconds)'][i:i + 2], self.sleep_stages['Sleep Stage Code'][i:i + 2], 
                            marker='o', linestyle='-', color=colors(self.sleep_stages['Sleep Stage Code'][i]), label=stage_label)
                    added_labels.add(stage_label)
                else:
                    plt.plot(self.sleep_stages['Time (seconds)'][i:i + 2], self.sleep_stages['Sleep Stage Code'][i:i + 2], 
                            marker='o', linestyle='-', color=colors(self.sleep_stages['Sleep Stage Code'][i]))

            plt.xlabel('Time (seconds)')
            plt.ylabel('Sleep Stage')
            plt.title('Sleep Stages Over Time')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.yticks(list(self.stage_mapping.keys()), list(self.stage_mapping.values()))
            plt.legend(title='Sleep Stages', loc='upper right', bbox_to_anchor=(1.2, 1))
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting sleep stages: {e}")
    
    def plot_sleep_stages_by_code(self):
        """
        Plot sleep stages over time by sleep stage code.
        """
        try:
            colors = plt.cm.get_cmap('tab10', len(self.stage_mapping))

            for stage_code, stage_label in self.stage_mapping.items():
                stage_times = self.sleep_stages[self.sleep_stages['Sleep Stage Code'] == stage_code]['Time (seconds)']
                stage_values = self.sleep_stages[self.sleep_stages['Sleep Stage Code'] == stage_code]['Sleep Stage Code']
                
                if not stage_times.empty:
                    plt.plot(stage_times, stage_values, marker='o', linestyle='-', 
                            color=colors(stage_code), label=stage_label)

            plt.xlabel('Time (seconds)')
            plt.ylabel('Sleep Stage')
            plt.title('Sleep Stages Over Time')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.yticks(list(self.stage_mapping.keys()), list(self.stage_mapping.values()))
            plt.legend(title='Sleep Stages', loc='upper right', bbox_to_anchor=(1.2, 1))
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting sleep stages by code: {e}")

    def analyze_sleep_stages(self):
        """
        Analyze the sleep stages and print statistics.
        """
        try:
            total_stages = Counter(self.sleep_stages['Sleep Stage Code'])
            total_on_bed_time = len(self.sleep_stages) * self.epoch_length

            def convert_to_min_sec(seconds):
                minutes = seconds // 60
                remaining_seconds = seconds % 60
                return minutes, remaining_seconds

            total_on_bed_minutes, total_on_bed_seconds = convert_to_min_sec(total_on_bed_time)
            print("Sleep Stage Analysis (including all stages):")
            print("=" * 30)

            for stage, count in total_stages.items():
                stage_time = count * self.epoch_length
                percentage = (stage_time / total_on_bed_time) * 100
                stage_label = self.stage_mapping[stage]
                print(f"{stage_label}: {stage_time} seconds ({percentage:.2f}%)")

            print("=" * 30)
            print(f"Total On-bed Time: {total_on_bed_minutes} minutes and {total_on_bed_seconds} seconds\n")

            excluded_stages = {0, 9}
            filtered_stages = {stage: count for stage, count in total_stages.items() if stage not in excluded_stages}
            total_sleep_time = sum(count for stage, count in filtered_stages.items()) * self.epoch_length
            total_sleep_minutes, total_sleep_seconds = convert_to_min_sec(total_sleep_time)

            print("Sleep Stage Analysis (excluding Wakefulness and Movement Time):")
            print("=" * 30)

            for stage, count in filtered_stages.items():
                stage_time = count * self.epoch_length
                percentage = (stage_time / total_sleep_time) * 100
                stage_label = self.stage_mapping[stage]
                print(f"{stage_label}: {stage_time} seconds ({percentage:.2f}%)")

            print("=" * 30)
            print(f"Total Sleep Time (excluding Wakefulness and Movement Time): {total_sleep_minutes} minutes and {total_sleep_seconds} seconds")
            
            percentage_sleep_over_on_bed = (total_sleep_time / total_on_bed_time) * 100
            print("=" * 30)
            print(f"Percentage of Sleep Time over On-bed Time: {percentage_sleep_over_on_bed:.2f}%")
        except Exception as e:
            print(f"Error analyzing sleep stages: {e}")



# Example usage:
if __name__ == "__main__":
    file_path = 'psg/20240620江逸凡.edf.XML'
    start_datetime_str = "2024-06-20 22:02:34"
    
    xml_processor = XMLProcessor(file_path, start_datetime_str)
    xml_processor.load()
    xml_processor.plot_sleep_stages()
    xml_processor.plot_sleep_stages_by_code()
    xml_processor.analyze_sleep_stages()

