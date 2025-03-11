# PATH = "/Users/w.z/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/SleepData"
# PATH = "C:/Users/amd/OneDrive - National University of Singapore/SleepData"


PATH = "/opt/data/private/ZhouWenren/SleepLab"

# 2024-06-20江逸凡
DATE_PATH_1 = "/psg/20240620"
EDF_FILE_PATH_1 = PATH + DATE_PATH_1 + "/20240620江逸凡.edf"
PSG_FILE_PATH_1 = PATH + DATE_PATH_1 + "/20240620江逸凡.edf"
XML_FILE_PATH_1 = PATH + DATE_PATH_1 + "/20240620江逸凡.edf.XML"
RADAR_FILE_PATH_1 = PATH + "/cw_radar/radar20240620220948433561.csv"


SAVE_PATH = PATH + "/psg/merged_2024-6-20jiangyifan.edf.pkl"
FULL_SAVE_PATH = PATH + "/psg/full_merged_2024-6-20jiangyifan.edf.pkl"


# 2024-09-25曹建侠
DATE_PATH_2 = "/psg/20240925"
FOLDER_PATH_2 = '/20240925/selected'
PSG_FILE_PATH_2 = PATH + DATE_PATH_2 + '/PSG2024092546曹建侠.edf'
XML_FILE_PATH_2 = PATH + DATE_PATH_2 + '/PSG2024092546曹建侠已分期.edf.XML'
RADAR_FILE_PATH_2 = PATH + DATE_PATH_2 + FOLDER_PATH_2 + '/adc_data_2024_09_25_Raw_LogFile.csv'
BREATH_PATH_2 = PATH + DATE_PATH_2 + '/20240925/range_0.8416breath_300s.csv'
HEART_PATH_2 = PATH + DATE_PATH_2 + '/20240925/range_0.8416heart_300s.csv'

# 2024-09-27俞兴男
DATE_PATH_3 = "/psg/20240927"
FOLDER_PATH_3 = '/20240927/selected'
PSG_FILE_PATH_3 = PATH + DATE_PATH_3 + '/PSG20240927俞兴男已分期.edf'
XML_FILE_PATH_3 = PATH + DATE_PATH_3 + '/PSG20240927俞兴男已分期.edf.XML'
RADAR_FILE_PATH_3 = PATH + DATE_PATH_3 + FOLDER_PATH_3 + '/adc_data_2024_09_27_Raw_LogFile.csv'
BREATH_PATH_3 = PATH + DATE_PATH_3 + '/20240927/0927breath.csv'
HEART_PATH_3 = PATH + DATE_PATH_3 + '/20240927/0927heart.csv'

# 2025-01-21李子银
DATE_PATH_4 = "/psg/20250121"
PSG_FILE_PATH_4 = PATH + DATE_PATH_4 + "/PSG20250101A李子银首夜仅PSG.edf"
XML_FILE_PATH_4 = PATH + DATE_PATH_4 + "/PSG20250101A李子银首夜仅PSG.edf.XML"
RADAR_FILE_PATH_4 = PATH + "/cw_radar/2025-01-21/radar20250121221903431329.csv"
# RADAR_FILE_PATH_4 = PATH + "/cw_radar/2025-01-21/radar20250122031639995741.csv"
SAVE_PATH_4 = PATH + "/psg/merged_2025-1-21liziyin.edf.pkl"
FULL_SAVE_PATH_4 = PATH + "/psg/full_merged_2025-1-21liziyin.edf.pkl"

# from pathlib import Path
# from dataclasses import dataclass
# from typing import Optional

# # Base path configuration
# BASE_PATH = Path("/opt/data/private/ZhouWenren/SleepLab")

# @dataclass
# class SubjectData:
#     date: str
#     name: str
#     psg_file: Path
#     xml_file: Path
#     radar_file: Optional[Path] = None
#     save_path: Optional[Path] = None
#     full_save_path: Optional[Path] = None

# class DataPaths:
#     def __init__(self, base_path: Path):
#         self.base_path = base_path
#         self.subjects = {}

#     def add_subject(self, date: str, name: str) -> SubjectData:
#         date_formatted = date.replace("-", "")
#         psg_dir = self.base_path / "psg" / date_formatted
        
#         subject = SubjectData(
#             date=date,
#             name=name,
#             psg_file=psg_dir / f"{date_formatted}{name}.edf",
#             xml_file=psg_dir / f"{date_formatted}{name}.edf.XML",
#             radar_file=self.base_path / "cw_radar" / f"radar{date_formatted}220948433561.csv",
#             save_path=self.base_path / "psg" / f"merged_{date}{name}.edf.pkl",
#             full_save_path=self.base_path / "psg" / f"full_merged_{date}{name}.edf.pkl"
#         )
#         self.subjects[f"{date}_{name}"] = subject
#         return subject

# # Initialize paths
# paths = DataPaths(BASE_PATH)

# # Add subjects
# SUBJECT_20240620 = paths.add_subject("2024-06-20", "江逸凡")
# SUBJECT_20240925 = paths.add_subject("2024-09-25", "曹建侠")

# # Usage example:
# # print(SUBJECT_20240620.psg_file)  # Access PSG file path
# # print(SUBJECT_20240925.xml_file)  # Access XML file path

# from constants import SUBJECT_20240620, SUBJECT_20240925