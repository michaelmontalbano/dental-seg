from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from typing import Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DentalRadiographPreprocessor(DefaultPreprocessor):
    """
    Custom preprocessor for 2D dental radiographs (panoramic, periapical, bitewing).
    Optimized for implant and dental feature detection.
    """

    def get_normalization_scheme(self) -> str:
        """
        Use z-score normalization for radiographs.
        X-rays don't have standardized intensity units like CT.
        """
        logger.info("📐 Using z-score normalization for dental radiographs")
        # Options: "ZScore", "CT", "noNorm"
        # Z-score is best for variable X-ray intensities across different machines
        return "ZScore"

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self) -> Tuple[bool, Tuple[float, ...], Tuple[int, ...], Tuple[int, ...]]:
        """
        Configure augmentation for 2D dental images.
        Disable left-right mirroring to preserve tooth numbering and anatomical side.
        """
        logger.info("🔁 Configuring augmentation for dental radiographs")
        
        # Get default configuration
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # For 2D images, we typically have axes: (height, width)
        # Disable left-right mirroring (axis 1) to preserve anatomical orientation
        # Only allow up-down mirroring (axis 0) if needed, though even this might be questionable
        mirror_axes = tuple()  
        # OR if you want only vertical mirroring (rarely useful for dental):
        # mirror_axes = (0,)  
        
        # For 2D radiographs, patch size should cover most of the image
        # Panoramic radiographs are typically wide (e.g., 2900x1500 pixels)
        # Periapical are more square (e.g., 1000x1200 pixels)
        # Set a reasonable patch size that works for both
        if len(initial_patch_size) == 2:  # 2D configuration
            # Use larger patches for radiographs to capture full dental context
            initial_patch_size = (512, 512)  # Adjust based on your image sizes
            
        logger.info(f"Mirror axes: {mirror_axes}, Patch size: {initial_patch_size}")
        
        return rotation_for_DA, do_dummy_2d_data_aug, mirror_axes, initial_patch_size

    def should_use_roi(self) -> bool:
        """
        Enable ROI cropping to remove black borders common in radiographs.
        This removes the circular/rectangular masks often present.
        """
        logger.info("📦 Enabling ROI-based cropping for border removal")
        return True

    def get_target_spacing(self) -> Tuple[float, ...]:
        """
        Set target spacing for resampling.
        For 2D radiographs, this is pixel spacing in mm.
        """
        # Most dental radiographs have spacing around 0.1-0.3mm per pixel
        # You might want to standardize to a common resolution
        target_spacing = (0.2, 0.2)  # in mm, adjust based on your dataset
        
        logger.info(f"📏 Target spacing for radiographs: {target_spacing} mm/pixel")
        return target_spacing

    def run_case_npy(self, data: np.ndarray, seg: np.ndarray, properties: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Additional preprocessing specific to dental radiographs.
        """
        # Call parent preprocessing first
        data, seg, properties = super().run_case_npy(data, seg, properties)
        
        # Optional: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This can improve contrast in radiographs
        # Uncomment if you want to use it:
        """
        from skimage import exposure
        for channel in range(data.shape[0]):
            # Normalize to 0-1 for CLAHE
            data_min = data[channel].min()
            data_max = data[channel].max()
            data_norm = (data[channel] - data_min) / (data_max - data_min + 1e-8)
            
            # Apply CLAHE
            data_clahe = exposure.equalize_adapthist(data_norm, clip_limit=0.03)
            
            # Scale back
            data[channel] = data_clahe * (data_max - data_min) + data_min
        """
        
        # Optional: Enhance contrast for implant detection
        # Implants are typically very bright in radiographs
        # You could apply specific enhancements here
        
        logger.info(f"✅ Preprocessed radiograph with shape: {data.shape}")
        
        return data, seg, properties

    def get_classes_for_oversampling(self) -> list:
        """
        Define which classes to oversample during training.
        Useful if implants are rare in your dataset.
        """
        # If you have class imbalance (e.g., few implants), list the rare class IDs here
        # Example: if implant class is label 1
        oversample_classes = [1]  # Adjust based on your label encoding
        
        logger.info(f"⚖️ Oversampling classes: {oversample_classes}")
        return oversample_classes

    def determine_normalization_params(self, data: np.ndarray, seg: np.ndarray = None) -> dict:
        """
        Calculate normalization parameters specifically for radiographs.
        """
        params = super().determine_normalization_params(data, seg)
        
        # Optional: Use percentile clipping for radiographs to handle outliers
        # This removes extreme bright/dark spots that might be artifacts
        percentile_00_5 = np.percentile(data, 0.5)
        percentile_99_5 = np.percentile(data, 99.5)
        
        params['percentile_00_5'] = percentile_00_5
        params['percentile_99_5'] = percentile_99_5
        
        logger.info(f"📊 Intensity range: [{percentile_00_5:.1f}, {percentile_99_5:.1f}]")
        
        return params