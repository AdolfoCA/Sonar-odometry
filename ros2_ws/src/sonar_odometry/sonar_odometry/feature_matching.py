import numpy as np
import cv2




class SonarFeatureMatcher:

    def __init__(self, lowe_ratio: float = 0.50):
        """
        Parameters
        ----------
        lowe_ratio : Lowe's ratio-test threshold for match filtering.
                     Lower values = stricter (fewer but more reliable matches).
        """
        self._lowe_ratio = lowe_ratio
        # Sonar geometry — populated from the first ROS message via update_sonar_params()
        self._beam_azimuths_rad: np.ndarray | None = None   # shape (beam_count,)
        self._ranges_m: np.ndarray | None = None             # shape (range_bins,)
        self._detector = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size=0,
            descriptor_channels=3,
            threshold=0.001,
            nOctaves=8,
            nOctaveLayers=8,
            diffusivity=cv2.KAZE_DIFF_PM_G2
        )

    def set_sonar_params(self, beam_azimuths_rad: np.ndarray, ranges_m: np.ndarray) -> None:
        """
        Set sonar geometry directly from numpy arrays.
        Useful for notebooks and unit tests where no ROS message is available.

        Parameters
        ----------
        beam_azimuths_rad : azimuth angle (rad) for each beam column, shape (n_beams,)
        ranges_m          : range (m) at each bin centre, shape (n_bins,)
        """
        self._beam_azimuths_rad = np.asarray(beam_azimuths_rad, dtype=np.float32)
        self._ranges_m          = np.asarray(ranges_m,          dtype=np.float32)

    def update_sonar_params(self, msg) -> None:
        """
        Extract and cache sonar geometry from a ProjectedSonarImage message.
        Called once on the first message; ignored on subsequent calls.

        Geometry convention (from marine_acoustic_msgs):
          - Z-axis forward (boresight), beams in Y-Z plane
          - Azimuth = rotation about X  →  atan2(beam_direction.y, beam_direction.z)
          - msg.ranges contains the actual range (m) of each bin centre
        """
        if self._beam_azimuths_rad is not None:
            return  # already cached

        if not msg.beam_directions or not msg.ranges:
            return  # message not yet populated

        self._beam_azimuths_rad = np.array(
            [np.arctan2(bd.y, bd.z) for bd in msg.beam_directions],
            dtype=np.float32,
        )
        self._ranges_m = np.array(msg.ranges, dtype=np.float32)

    def polar_to_cartesian_coords(self, col: float, row: float) -> np.ndarray:
        """
        Convert polar sonar image coordinates to Cartesian (metres).

        Uses the geometry cached from the ROS message via update_sonar_params().
        Sub-pixel (col, row) values are handled with linear interpolation.

        Returns [x, y]:  x = forward (boresight),  y = right (positive starboard)
        """
        if self._beam_azimuths_rad is None or self._ranges_m is None:
            raise RuntimeError(
                "Sonar geometry not set. Call update_sonar_params(msg) before matching."
            )

        col_idx = np.arange(len(self._beam_azimuths_rad), dtype=np.float32)
        row_idx = np.arange(len(self._ranges_m),          dtype=np.float32)

        azimuth = float(np.interp(col, col_idx, self._beam_azimuths_rad))
        range_m = float(np.interp(row, row_idx, self._ranges_m))

        x = range_m * np.cos(azimuth)    # forward
        y = -range_m * np.sin(azimuth)  # right (positive starboard); negate because beam_directions.y is port-positive in marine_acoustic_msgs
        return np.array([x, y], dtype=np.float32)

######################## AKAZE ###############################
    

    def detect_sonar_features(self, image):
        """
        Detect features optimized for sonar images using AKAZE.
        """
        
        keypoints, descriptors = self._detector.detectAndCompute(image, None)
        
        return keypoints, descriptors

    def match_sonar_features(self, desc1, desc2, distance_threshold=None):
        """
        Match features using BFMatcher with Hamming distance for MLDB_UPRIGHT.
        """
        threshold = distance_threshold if distance_threshold is not None else self._lowe_ratio

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
            if m.distance < threshold * n.distance:
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
        #TODO: I also want to set this threshold in the config file.
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
    

