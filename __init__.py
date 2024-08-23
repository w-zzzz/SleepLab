# SleepLab/__init__.py

# Importing specific utilities to make them accessible directly from the SleepLab package
from .utils import resample_data
from .constants import *

# You can also import specific classes or functions from your submodules if needed
# from .data_extraction import some_function_or_class

# If you want to include the submodules as part of the SleepLab package
import SleepLab.cw_radar
import SleepLab.psg

# Defining what should be available for import when using 'from SleepLab import *'
__all__ = [ 'resample_data', 'cw_radar', 'psg']
