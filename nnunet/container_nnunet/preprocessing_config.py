"""
Custom preprocessing configuration for dental CBCT data
"""

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from typing import Tuple

class DentalCBCTPreprocessor(DefaultPreprocessor):
    """Custom preprocessor for dental CBCT data"""
    
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self) -> Tuple[bool, Tuple[float, ...], bool, Tuple[int, ...]]:
        """
        Override to disable left-right mirroring for dental data
        """
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # Disable mirroring on the left-right axis (usually axis 0)
        # This is critical for tooth identification
        mirror_axes = (1, 2)  # Only allow mirroring on other axes
        
        return rotation_for_DA, do_dummy_2d_data_aug, mirror_axes, initial_patch_size
    
    def get_normalization_scheme(self) -> str:
        """
        Use CT normalization for CBCT data
        """
        return "CT"
