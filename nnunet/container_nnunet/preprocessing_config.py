from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from typing import Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DentalCBCTPreprocessor(DefaultPreprocessor):
    """
    Custom preprocessor for dental CBCT data.
    Disables left-right mirroring and configures dental-specific behavior.
    """

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self) -> Tuple[bool, Tuple[float, ...], Tuple[int, ...], Tuple[int, ...]]:
        """
        Override to disable left-right mirroring and adjust patch size if needed.
        """
        logger.info("🔁 Overriding mirroring configuration for dental CBCT")
        
        # Get default configuration
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # Disable mirroring on left-right axis (usually axis 0)
        mirror_axes = (1, 2)  # Allow mirroring on height/width only

        # Optional: manually set a more stable patch size for large-volume CBCT
        # initial_patch_size = (192, 192, 96)

        return rotation_for_DA, do_dummy_2d_data_aug, mirror_axes, initial_patch_size

    def get_normalization_scheme(self) -> str:
        """
        Override normalization to use CT standard.
        """
        logger.info("📐 Using CT normalization for dental CBCT")
        return "CT"

    def should_use_roi(self) -> bool:
        """
        Enable ROI cropping to remove slices with only background.
        """
        logger.info("📦 Enabling ROI-based cropping")
        return True

    def get_spacings(self) -> Tuple[float, float, float]:
        """
        Optionally override target spacing.
        Useful when CBCT has highly variable resolution.
        """
        default_spacing = super().get_spacings()
        # Uncomment and edit to enforce consistent spacing
        # default_spacing = (0.3, 0.3, 0.3)
        logger.info(f"📏 Using target spacing: {default_spacing}")
        return default_spacing

    def get_class_balancing(self) -> bool:
        """
        Toggle class balancing during sampling.
        If False, relies on natural class frequency.
        """
        use_balancing = True  # Set to False if you don't want oversampling of rare labels
        logger.info(f"⚖️ Class balancing: {'enabled' if use_balancing else 'disabled'}")
        return use_balancing
