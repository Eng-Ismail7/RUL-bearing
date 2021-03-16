from enum import Enum
from util.constants import SPECTRA_CSV_NAME, FEATURES_CSV_NAME, RAW_CSV_NAME


class DataSetType(Enum):
    spectra = SPECTRA_CSV_NAME
    computed = FEATURES_CSV_NAME
    raw = RAW_CSV_NAME
