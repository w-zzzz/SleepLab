# PATH = "/Users/w.z/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/SleepData"
# PATH = "C:/Users/amd/OneDrive - National University of Singapore/SleepData"


PATH = "/opt/data/private/ZhouWenren/SleepLab"

# 2024-06-20江逸凡
PSG_PATH_1 = "/psg/data/2024-06-20"
RADAR_PATH_1 = "/cw_radar/data/2024-06-20"
PSG_FILE_PATH_1 = PATH + PSG_PATH_1 + "/20240620江逸凡.edf"
XML_FILE_PATH_1 = PATH + PSG_PATH_1 + "/20240620江逸凡.edf.XML"
RADAR_FILE_PATH_1 = PATH + RADAR_PATH_1 + "radar20240620220948433561.csv"

# 2024-09-25曹建侠
PSG_PATH_2 = "/psg/data/2024-09-25"
RADAR_PATH_2 = "/cw_radar/data/2024-09-25"
PSG_FILE_PATH_2 = PATH + PSG_PATH_2 + '/PSG2024092546曹建侠.edf'
XML_FILE_PATH_2 = PATH + PSG_PATH_2 + '/PSG2024092546曹建侠已分期.edf.XML'
RADAR_FILE_PATH_2 = PATH + RADAR_PATH_2 + '/adc_data_2024_09_25_Raw_LogFile.csv'
BREATH_PATH_2 = PATH + RADAR_PATH_2 + '/20240925/range_0.8416breath_300s.csv'
HEART_PATH_2 = PATH + RADAR_PATH_2 + '/20240925/range_0.8416heart_300s.csv'

# 2024-09-27俞兴男
PSG_PATH_3 = "/psg/data/2024-09-27"
RADAR_PATH_3 = '/cw_radar/data/2024-09-27'
PSG_FILE_PATH_3 = PATH + PSG_PATH_3 + '/PSG20240927俞兴男已分期.edf'
XML_FILE_PATH_3 = PATH + PSG_PATH_3 + '/PSG20240927俞兴男已分期.edf.XML'
RADAR_FILE_PATH_3 = PATH + RADAR_PATH_3 + '/adc_data_2024_09_27_Raw_LogFile.csv'
BREATH_PATH_3 = PATH + RADAR_PATH_3 + '/0927breath.csv'
HEART_PATH_3 = PATH + RADAR_PATH_3 + '/0927heart.csv'

# 2025-01-21李子银
PSG_PATH_4 = "/psg/data/2025-01-21"
RADAR_PATH_4 = "/cw_radar/data/2025-01-21"
PSG_FILE_PATH_4 = PATH + PSG_PATH_4 + "/PSG20250101A李子银首夜仅PSG.edf"
XML_FILE_PATH_4 = PATH + PSG_PATH_4 + "/PSG20250101A李子银首夜仅PSG.edf.XML"
RADAR_FILE_PATH_4 = PATH + RADAR_PATH_4 + "/radar20250121221903431329.csv"
RADAR_FILE_PATH_4_2nd = PATH + "/cw_radar/2025-01-21/radar20250122031639995741.csv"


# 2025-02-26任红兵
PSG_PATH_5 = "/psg/data/2025-02-26"
RADAR_PATH_5 = "/cw_radar/data/2025-02-26"
PSG_FILE_PATH_5 = PATH + PSG_PATH_5 + "/PSG2025022631A任红兵.edf"
XML_FILE_PATH_5 = PATH + PSG_PATH_5 + "/PSG2025022631A任红兵.edf.XML"
RADAR_FILE_PATH_5 = PATH + RADAR_PATH_5 + "/radar20250226233910744239.csv"
RADAR_FILE_PATH_5_2nd = PATH + RADAR_PATH_5 + "/radar20250226222805302380.csv"

# 2025-02-27任红兵
PSG_PATH_6 = "/psg/data/2025-02-27"
RADAR_PATH_6 = "/cw_radar/data/2025-02-27"
PSG_FILE_PATH_6 = PATH + PSG_PATH_6 + "/20250227E任红兵.edf"
XML_FILE_PATH_6 = PATH + PSG_PATH_6 + "/20250227E任红兵.edf.XML"
RADAR_FILE_PATH_6 = PATH + RADAR_PATH_6 + "/radar20250227225543092087.csv"


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