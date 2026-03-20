import numpy as np
import cv2


def get_default_sonar_params():
        """Default sonar parameters from your data"""
        return {
            'frequency': 749001.3125,
            'sound_speed': 1480.033935546875,
            'beam_count': 512,
            'azimuth_min_deg': -65.0,
            'azimuth_max_deg': 65.0,
            'azimuth_span_deg': 130.0,
            'beam_spacing_deg': 0.2544031383028491,
            'range_bins': 633,
            'min_range_m': 0.007893514,
            'max_range_m': 9.985295,
            'range_resolution_m': 0.015787028,
            'is_uniform_beams': False,
            'is_uniform_range': True
        }



class SonarFeatureMatcher:

    def __init__(self):
        self._detector = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size=0,
            descriptor_channels=3,
            threshold=0.0001,
            nOctaves=8,
            nOctaveLayers=8,
            diffusivity=cv2.KAZE_DIFF_PM_G2
        )

    ## This returns the transofrmation estimation between two sonar images
    def polar_to_cartesian_coords(self, col: float, row: float) -> np.ndarray:
        """
        Convert polar sonar coordinates to Cartesian coordinates.
        
        Args:
            col: Column index (azimuth/beam index)
            row: Row index (range bin)
            sonar_params: Sonar configuration parameters
            
        Returns:
            Cartesian coordinates [x, y] in meters
        """
        sonar_params = get_default_sonar_params()
        # Extract sonar parameters
        azimuth_span_deg = sonar_params.get('azimuth_span_deg')
        azimuth_min_deg = sonar_params.get('azimuth_min_deg')
        beam_count = sonar_params.get('beam_count')
        min_range_m = sonar_params.get('min_range_m')
        max_range_m = sonar_params.get('max_range_m')
        range_bins = sonar_params.get('range_bins')
        beam_spacing_deg = 0.2544
        range_resolution_m = 0.015787028
        
        # Calculate angle for this beam
        angle_deg = azimuth_min_deg + (col * beam_spacing_deg)
        angle_rad = np.deg2rad(angle_deg)
        
        # Calculate range for this bin
        range_m = min_range_m + (row * range_resolution_m)
        
        # Convert to Cartesian coordinates
        x = range_m * np.cos(angle_rad)
        y = range_m * np.sin(angle_rad)
        
        return np.array([x, y], dtype=np.float32)
        # X: forward 
        # Y: right

######################## AKAZE ###############################
    

    def detect_sonar_features(self, image):
        """
        Detect features optimized for sonar images using AKAZE.
        """
        
        keypoints, descriptors = self._detector.detectAndCompute(image, None)
        
        return keypoints, descriptors

    def match_sonar_features(self, desc1, desc2, distance_threshold=0.50): # 0.50 is a good one
        """
        Match features using BFMatcher with Hamming distance for MLDB_UPRIGHT.
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        
        # Use BFMatcher with HAMMING distance for binary descriptors
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Find 2 nearest neighbors for ratio test
        knn_matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance < distance_threshold * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def estimate_transformation(self, kp1, kp2, matches):
        """
        Estimate transformation between matched keypoints.
        """
        if len(matches) < 4:
            return {
                "transformation": None,
                "inliers": None,
                "num_inliers": 0,
                "inlier_ratio": 0.0
            }
        
        # Extract matched points (these are in polar image coordinates)
        pts1_polar = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2_polar = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Convert to Cartesian coordinates
        pts1_cart = np.float32([
            self.polar_to_cartesian_coords(col, row) 
            for (col, row) in pts1_polar
        ])
        pts2_cart = np.float32([
            self.polar_to_cartesian_coords(col, row) 
            for (col, row) in pts2_polar
        ])
        
        # Use meter-based threshold for Cartesian coordinates
        ranges = np.linalg.norm(pts1_cart, axis=1)
        #reproj_threshold_m = max(0.05, np.median(ranges) * 0.03)  # 3% of median range
        #print (f"Reprojection threshold (m): {reproj_threshold_m}")
        #reproj_threshold_m = 0.1 # 1 meter
        reproj_threshold_m = 0.5 # 2 meters
        
        # Estimate transformation in Cartesian space
        # I2 = T * I1
        transformation_image, inliers = cv2.estimateAffinePartial2D( # from image 1 to image 2 
            pts1_cart, pts2_cart,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj_threshold_m,
            maxIters=5000,
            confidence=0.99,
            refineIters=200
        )
        # from S1 to S2 
        #transformation = transformation_image
        transformation = cv2.invertAffineTransform(transformation_image)
        if transformation is not None:
            # Extract and normalize rotation matrix
            rotation_matrix = transformation[:2, :2]
            translation = transformation[:2, 2]
                     
            # Force unit scale
            u, s, vt = np.linalg.svd(rotation_matrix)
            transformation[:2, :2] = u @ vt

        if transformation is None or inliers is None:
            return {
                "transformation": None,
                "inliers": None,
                "num_inliers": 0,
                "inlier_ratio": 0.0
            }
        
        num_inliers = int(np.sum(inliers))
        inlier_ratio = num_inliers / len(matches)
        
        return {
            "transformation": transformation,
            "inliers": inliers.ravel().astype(bool),
            "num_inliers": num_inliers,
            "inlier_ratio": inlier_ratio
        }
    

    def process_sonar_image_pair(self, img1, img2, timestamp=None):
        """
        Process a pair of consecutive sonar images to estimate motion.
        
        Args:
            img1: Previous sonar image
            img2: Current sonar image
            timestamp: Current timestamp (optional)
            visualize: If True, display feature matches (default: False)
        
        Returns:
            Dictionary with processing results
        """

        # Detect sonar-optimized features
        kp1, desc1 = self.detect_sonar_features(img1)
        kp2, desc2 = self.detect_sonar_features(img2)
        matches = self.match_sonar_features(desc1, desc2)
        
        # Estimate transformation
        result = self.estimate_transformation(kp1, kp2, matches)
                
        # Update trajectory if transformation is valid and has sufficient inliers
        min_inliers = 6  # Minimum inliers for reliable sonar odometry
        if (result['transformation'] is not None and
            result['num_inliers'] >= min_inliers and
            result['inlier_ratio'] > 0.3):  # At least 30% inlier ratio
            # debug
            #print(f"number inliers: {result['num_inliers']}")
            pass
        
        return result, kp1,kp2,matches
    

