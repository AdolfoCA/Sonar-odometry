import numpy as np
import cv2


class SonarImageProcessor:
    def __init__(self):
        self.config = {
            "apply_denoise": True, # always true
            "apply_gaussian": True,
            "apply_median": True,
            "apply_otsu": False, # always false
            "apply_fuzzy": True,
            "apply_opening": True,
            "gamma_value": 1.6,
            "gaussian_ksize": (3, 3),
            "gaussian_sigma": 0,
            "median_ksize": 3,
            "morph_kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            "final_normalization": False
        }
    
    def apply_denoising(self, img: np.ndarray) -> np.ndarray:
        """
        Apply row-wise background subtraction denoising.
        
        Args:
            img: Input image
            
        Returns:
            Denoised image
        """
        if not self.config.get("apply_denoise", False):
            return img
        
        try:
            # Ensure img is float for processing
            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255.0 if img.max() > 1 else img.astype(np.float32)
            
            # Estimate background using quantile per row
            background = np.quantile(img, 0.1, axis=1, keepdims=True)
            img_denoised = np.clip(img - background, 0.0, None)
            
            # Renormalize if image has content
            if img_denoised.max() > 0:
                img_denoised = img_denoised / (img_denoised.max() + 1e-6)
            
            return img_denoised
        
        except Exception as e:
            #print(f"Warning: Denoising failed: {e}")
            return img
    
    def apply_smoothing(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian and median smoothing filters.
        
        Args:
            img: Input image
            
        Returns:
            Smoothed image
        """
        # Gaussian blur
        if self.config.get("apply_gaussian", True):
            img = cv2.GaussianBlur(
                img, 
                self.config["gaussian_ksize"], 
                self.config["gaussian_sigma"]
            )
        
        # Median filter
        if self.config.get("apply_median", True):
            img_u8 = (img * 255).astype(np.uint8)
            img_median = cv2.medianBlur(img_u8, self.config["median_ksize"])
            img = img_median.astype(np.float32) / 255.0
        
        return img
    
    def apply_thresholding(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Otsu thresholding for binary masking using OpenCV.
        
        Args:
            img: Input image
            
        Returns:
            Thresholded image
        """
        if not self.config.get("apply_otsu", False):
            return img
        
        try:
            img_u8 = (img * 255).astype(np.uint8)
            # Use OpenCV's Otsu thresholding instead of skimage
            ret, binary_mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_mask = binary_mask.astype(np.float32) / 255.0
            return img * binary_mask
        
        except Exception as e:
            #print(f"Warning: Otsu thresholding failed: {e}")
            return img
    
    def apply_enhancement(self, img: np.ndarray) -> np.ndarray:
        """
        Apply gamma correction and fuzzy enhancement.
        
        Args:
            img: Input image
            
        Returns:
            Enhanced image
        """
        if not self.config.get("apply_fuzzy", False):
            return img
        
        try:
            epsilon = 1e-6
            img = np.clip(img, 0.0, 1.0)
            gamma = float(self.config.get("gamma_value", 2.0))
            
            # Fuzzy gamma enhancement
            img_enhanced = (img ** gamma) / (img ** gamma + (1.0 - img + epsilon) ** gamma)
            return img_enhanced
        
        except Exception as e:
            #print(f"Warning: Enhancement failed: {e}")
            return img
    
    def apply_morphological_operations(self, img: np.ndarray) -> np.ndarray:
        """
        Apply morphological opening for noise removal.
        
        Args:
            img: Input image
            
        Returns:
            Morphologically processed image
        """
        if not self.config.get("apply_opening", False):
            return img
        
        try:
            img_u8 = (img * 255).astype(np.uint8)
            img_opened = cv2.morphologyEx(
                img_u8, 
                cv2.MORPH_OPEN, 
                self.config["morph_kernel"]
            )
            return img_opened.astype(np.float32) / 255.0
        
        except Exception as e:
            #print(f"Warning: Morphological operations failed: {e}")
            return img
    
    def threshold_image(self, img: np.ndarray, thresh: float = 0.50) -> np.ndarray:
        """
        Apply simple thresholding to create a binary mask.
        
        Args:
            img: Input image
            thresh: Threshold value
            
        Returns:
            Binary masked image
        """
        max = img.max()
        thresh = max * thresh
        #print(f"Image max value before thresholding: {max}")
        return (img > thresh).astype(np.float32) 
    
    def apply_final_normalization(self, img: np.ndarray) -> np.ndarray:
        """
        Apply final normalization to ensure [0,1] range.
        
        Args:
            img: Input image
            
        Returns:
            Normalized image
        """
        if not self.config.get("final_normalization", True):
            return img
        
        return cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX)
    
    def size_image(self, img: np.ndarray) -> np.ndarray:
        """
        Set pixels to zero for rows 0-50 and last 50 rows of the image.
        
        Args:
            img: Input sonar image (numpy array)
        
        Returns:
            Modified image with top and bottom regions set to zero
        """
        height = img.shape[0]
        new_image = img.copy()
        
        # Set top 50 rows to zero
        new_image[0:100, :] = 0
        
        # Set bottom 50 rows to zero
        new_image[height-50:height, :] = 0
        
        return new_image

    
    def process_image(self, img: np.ndarray) -> np.ndarray:
        
        # Apply denoising
        if self.config.get("apply_denoise", False):
            img = self.apply_denoising(img)
        
        # Apply smoothing
        if self.config.get("apply_gaussian", False) or self.config.get("apply_median", False):
            ##print("Applying smoothing...")
            img = self.apply_smoothing(img)
        
        # Apply thresholding
        if self.config.get("apply_otsu", False):
            ##print("Applying Otsu thresholding...")
            img = self.apply_thresholding(img)
            ##print(f"Image size after Otsu thresholding: {img.shape}")
        
        # Apply enhancement
        if self.config.get("apply_fuzzy", False):
            img = self.apply_enhancement(img)
        
        # Apply morphological operations
        if self.config.get("apply_opening", False):
            img = self.apply_morphological_operations(img)
        # Apply final normalization
        if self.config.get("final_normalization", True):
            img = self.apply_final_normalization(img)
            
        img = self.size_image(img)
        img = self.threshold_image(img, thresh=0.20)

        return img
