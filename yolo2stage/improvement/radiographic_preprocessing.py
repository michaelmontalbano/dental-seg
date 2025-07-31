"""
Radiographic Quality Enhancement for Dental X-rays
Specialized preprocessing for periapical and bitewing radiographs
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Union, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DentalRadiographicEnhancer:
    """
    Specialized preprocessing for dental radiographic images.
    Optimized for periapical and bitewing X-rays.
    """
    
    def __init__(self, 
                 apply_clahe: bool = True,
                 apply_denoising: bool = True,
                 apply_histogram_eq: bool = True,
                 clahe_clip_limit: float = 3.0,
                 clahe_tile_size: int = 8,
                 denoise_strength: int = 10,
                 bilateral_d: int = 9,
                 bilateral_sigma_color: int = 75,
                 bilateral_sigma_space: int = 75):
        """
        Initialize radiographic enhancer.
        
        Args:
            apply_clahe: Whether to apply CLAHE enhancement
            apply_denoising: Whether to apply noise reduction
            apply_histogram_eq: Whether to apply histogram equalization
            clahe_clip_limit: Contrast limiting parameter for CLAHE
            clahe_tile_size: Grid size for CLAHE
            denoise_strength: Strength of denoising (1-30)
            bilateral_d: Diameter for bilateral filter
            bilateral_sigma_color: Filter sigma in color space
            bilateral_sigma_space: Filter sigma in coordinate space
        """
        self.apply_clahe = apply_clahe
        self.apply_denoising = apply_denoising
        self.apply_histogram_eq = apply_histogram_eq
        
        # CLAHE parameters
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_size, clahe_tile_size)
        )
        
        # Denoising parameters
        self.denoise_strength = denoise_strength
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
    
    def enhance_radiograph(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Apply full radiographic enhancement pipeline.
        
        Args:
            image: Input radiograph (numpy array or PIL Image)
            
        Returns:
            Enhanced radiograph as numpy array
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Store original for blending
        original = image.copy()
        
        # Step 1: Noise reduction (applied first to avoid amplifying noise)
        if self.apply_denoising:
            image = self._reduce_radiographic_noise(image)
        
        # Step 2: Histogram equalization (global contrast)
        if self.apply_histogram_eq:
            image = self._adaptive_histogram_equalization(image)
        
        # Step 3: CLAHE (local contrast)
        if self.apply_clahe:
            image = self._apply_clahe_enhancement(image)
        
        # Step 4: Edge preservation blend
        # Blend with original to preserve fine anatomical details
        image = cv2.addWeighted(image, 0.8, original, 0.2, 0)
        
        return image
    
    def _reduce_radiographic_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction specific to radiographic patterns.
        Combines multiple techniques for optimal results.
        """
        # Non-local means denoising (excellent for X-ray noise patterns)
        denoised = cv2.fastNlMeansDenoising(
            image,
            h=self.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Bilateral filter to preserve edges
        bilateral = cv2.bilateralFilter(
            denoised,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
        
        # Weighted combination
        result = cv2.addWeighted(denoised, 0.6, bilateral, 0.4, 0)
        
        return result
    
    def _adaptive_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive histogram equalization for global contrast.
        Includes special handling for over/under-exposed regions.
        """
        # Identify over/under-exposed regions
        very_dark = image < 20
        very_bright = image > 235
        
        # Apply standard histogram equalization
        equalized = cv2.equalizeHist(image)
        
        # Preserve extreme values (prevent washout)
        equalized[very_dark] = image[very_dark]
        equalized[very_bright] = image[very_bright]
        
        # Smooth transitions
        kernel = np.ones((3, 3), np.float32) / 9
        equalized = cv2.filter2D(equalized, -1, kernel)
        
        return equalized
    
    def _apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE with radiograph-specific parameters.
        """
        # Apply CLAHE
        enhanced = self.clahe.apply(image)
        
        # Post-process to reduce artifacts in uniform regions
        # Use edge detection to identify regions that should be preserved
        edges = cv2.Canny(image, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # Blend CLAHE result more strongly in edge regions
        mask = edges.astype(np.float32) / 255.0
        result = enhanced * mask + image * (1 - mask)
        
        return result.astype(np.uint8)
    
    def preprocess_for_model(self, image: Union[np.ndarray, Image.Image],
                           target_size: Tuple[int, int] = (224, 224),
                           normalize: bool = True) -> torch.Tensor:
        """
        Complete preprocessing pipeline for model input.
        
        Args:
            image: Input radiograph
            target_size: Target size for model
            normalize: Whether to apply ImageNet normalization
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Apply radiographic enhancement
        enhanced = self.enhance_radiograph(image)
        
        # Convert to RGB (most models expect 3 channels)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL for torchvision transforms
        pil_image = Image.fromarray(enhanced_rgb)
        
        # Define transform pipeline
        transform_list = [
            transforms.Resize(target_size),
            transforms.ToTensor()
        ]
        
        if normalize:
            # Modified normalization for radiographs
            # Less aggressive than ImageNet stats
            transform_list.append(
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],  # Centered normalization
                    std=[0.25, 0.25, 0.25]  # Preserve more range
                )
            )
        
        transform = transforms.Compose(transform_list)
        return transform(pil_image)
    
    def batch_enhance(self, images: list) -> list:
        """
        Enhance a batch of radiographs.
        
        Args:
            images: List of images (numpy arrays or PIL Images)
            
        Returns:
            List of enhanced images
        """
        return [self.enhance_radiograph(img) for img in images]
    
    def get_enhancement_params(self) -> Dict:
        """Get current enhancement parameters."""
        return {
            'apply_clahe': self.apply_clahe,
            'apply_denoising': self.apply_denoising,
            'apply_histogram_eq': self.apply_histogram_eq,
            'clahe_clip_limit': self.clahe_clip_limit,
            'clahe_tile_size': self.clahe_tile_size,
            'denoise_strength': self.denoise_strength,
            'bilateral_d': self.bilateral_d,
            'bilateral_sigma_color': self.bilateral_sigma_color,
            'bilateral_sigma_space': self.bilateral_sigma_space
        }


class AdaptiveRadiographicEnhancer(DentalRadiographicEnhancer):
    """
    Advanced version that adapts parameters based on image characteristics.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def analyze_image_quality(self, image: np.ndarray) -> Dict:
        """
        Analyze radiograph quality metrics.
        
        Returns:
            Dictionary with quality metrics
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate metrics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Entropy (measure of information content)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Contrast (using Michelson contrast)
        i_max = gray.max()
        i_min = gray.min()
        contrast = (i_max - i_min) / (i_max + i_min + 1e-10)
        
        # Noise estimation (using Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = laplacian.var()
        
        return {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'entropy': entropy,
            'contrast': contrast,
            'noise_level': noise_level,
            'is_underexposed': mean_intensity < 50,
            'is_overexposed': mean_intensity > 200,
            'is_low_contrast': contrast < 0.3,
            'is_noisy': noise_level > 500
        }
    
    def enhance_radiograph(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Apply adaptive enhancement based on image characteristics.
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Analyze image
        metrics = self.analyze_image_quality(image)
        
        # Adapt parameters based on analysis
        if metrics['is_underexposed']:
            self.clahe_clip_limit = 4.0  # More aggressive CLAHE
            logger.info("Detected underexposed image, increasing CLAHE limit")
        
        if metrics['is_low_contrast']:
            self.clahe_tile_size = 6  # Smaller tiles for more local enhancement
            logger.info("Detected low contrast, reducing tile size")
        
        if metrics['is_noisy']:
            self.denoise_strength = min(20, self.denoise_strength * 1.5)
            logger.info("Detected high noise, increasing denoising strength")
        
        # Apply standard enhancement
        enhanced = super().enhance_radiograph(image)
        
        # Reset parameters
        self.__init__(**self.get_enhancement_params())
        
        return enhanced


def create_dental_preprocessor(mode: str = 'standard') -> DentalRadiographicEnhancer:
    """
    Factory function to create appropriate preprocessor.
    
    Args:
        mode: 'standard', 'adaptive', 'light', or 'aggressive'
        
    Returns:
        Configured preprocessor instance
    """
    if mode == 'standard':
        return DentalRadiographicEnhancer()
    
    elif mode == 'adaptive':
        return AdaptiveRadiographicEnhancer()
    
    elif mode == 'light':
        # Lighter processing for good quality images
        return DentalRadiographicEnhancer(
            clahe_clip_limit=2.0,
            denoise_strength=5,
            apply_histogram_eq=False
        )
    
    elif mode == 'aggressive':
        # Heavy processing for poor quality images
        return DentalRadiographicEnhancer(
            clahe_clip_limit=4.0,
            clahe_tile_size=6,
            denoise_strength=15,
            bilateral_d=15
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


# Example usage in dataset
class EnhancedDentalDataset(torch.utils.data.Dataset):
    """Example dataset using radiographic enhancement."""
    
    def __init__(self, image_paths: list, labels: list, 
                 enhancer: Optional[DentalRadiographicEnhancer] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.enhancer = enhancer or DentalRadiographicEnhancer()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx])
        
        # Apply enhancement and preprocessing
        tensor = self.enhancer.preprocess_for_model(image)
        
        return tensor, self.labels[idx]
