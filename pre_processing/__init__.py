import sys
sys.path.append("./")

from util.constants import LEARNING_SET, FULL_TEST_SET
from pre_processing.features import compute_feature_data_frame
from pre_processing.spectra_features import compute_spectra_all_bearings

if __name__ == '__main__':
    """
    Select a data set sub set that should be processed.
    All bearings from that subset will be processed and stored in data set subset path out.
    """
    compute_spectra_all_bearings(LEARNING_SET)
    compute_spectra_all_bearings(FULL_TEST_SET)
    compute_feature_data_frame(LEARNING_SET)
    compute_feature_data_frame(FULL_TEST_SET)
