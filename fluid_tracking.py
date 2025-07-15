import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import json
import pickle
from scipy import optimize
from scipy.interpolate import UnivariateSpline
import random

class FluidTracker:
    def __init__(self, video_path, output_path=None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.output_path = output_path
        self.surface_history = deque(maxlen=100)  # Increased history for better analysis
        self.motion_vectors = []
        self.surface_positions = []  # Track surface position over time
        
        # Dynamic color detection parameters - will be set by user selection
        self.target_color_bgr = None
        self.target_color_hsv = None
        self.adaptive_hsv_ranges = None
        
        # Default color detection parameters for specific pink fluid (RGB: 200, 79, 114)
        # These will be overridden by dynamic selection
        self.lower_pink1 = np.array([160, 40, 50])   # Lower HSV range for pink (330-360° in H)
        self.upper_pink1 = np.array([180, 255, 255]) # Upper HSV range for pink
        self.lower_pink2 = np.array([0, 40, 50])     # Lower HSV range for pink (0-20° in H)
        self.upper_pink2 = np.array([20, 255, 255])  # Upper HSV range for pink
        
        # Adaptive color detection parameters
        self.color_tolerance = 40  # Increased tolerance for better adaptability
        self.min_saturation = 30   # Reduced minimum saturation for varied lighting
        self.min_value = 30        # Reduced minimum value for darker scenes
        self.hue_tolerance = 20    # Tolerance for hue variations
        self.saturation_tolerance = 60  # Tolerance for saturation variations
        self.value_tolerance = 80  # Tolerance for value variations
        
        # Color selection variables
        self.color_selected = False
        self.color_samples = []
        self.sample_region_size = 20  # Size of sampling region around clicked point
        
        # Morphological operations parameters
        self.kernel_size = 3  # Reduced kernel size for more precise detection
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        
        # Parameters for surface detection
        self.canny_low = 50
        self.canny_high = 150
        self.roi_top = 0.1  # ROI for surface detection
        self.roi_bottom = 0.9
        
        # Optical flow parameters
        self.lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Feature detection parameters
        self.feature_params = dict(maxCorners=100,
                                  qualityLevel=0.3,
                                  minDistance=10,
                                  blockSize=7)
        
        # Manual ROI selection
        self.selected_roi = None
        self.roi_points = []
        self.selecting = False
        self.roi_selected = False
        
        # Surface tracking parameters
        self.min_contour_area = 500   # Reduced minimum area for better adaptability
        self.smoothing_window = 7     # Increased window for better smoothing
        self.max_surface_jump = 50    # Maximum allowed jump in surface position
        
        # Container edge filtering
        self.edge_margin = 20         # Margin from image edges to ignore
        self.min_fluid_width = 50     # Reduced minimum width for better adaptability
        
        # Color profile management
        self.color_profiles_dir = "color_profiles"
        if not os.path.exists(self.color_profiles_dir):
            os.makedirs(self.color_profiles_dir)
        
        # Robust surface detection parameters - improved for fast movements
        self.use_robust_detection = True
        self.ransac_threshold = 15.0        # Increased for fast movements
        self.ransac_min_samples = 8         # Reduced for more flexibility
        self.ransac_iterations = 150        # Increased for better results
        self.min_inlier_ratio = 0.25        # Reduced for more tolerance
        
        # Temporal smoothing parameters - improved for fast movements
        self.use_temporal_smoothing = True
        self.temporal_weight = 0.7          # Weight for temporal consistency
        self.max_temporal_jump = 60         # Increased for fast movements
        self.adaptive_temporal_jump = True  # Enable adaptive temporal jump based on motion
        
        # Physics-based constraints - relaxed for fast movements
        self.max_surface_slope = 0.4        # Increased for fast movements
        self.smoothness_weight = 0.6        # Reduced for more flexibility
        self.gravity_direction = 1          # 1 for downward, -1 for upward
        
        # Kalman filter for temporal tracking - improved for fast movements
        self.kalman_filters = {}            # One filter per x-coordinate
        self.prev_surface_points = None     # Previous frame surface points
        self.surface_velocity = None        # Surface velocity estimate
        self.motion_speed = 0.0             # Current motion speed estimate
        self.fast_motion_threshold = 20     # Threshold for fast motion detection
        
        # Surface point density parameters
        self.surface_point_density = 2      # Points per pixel for dense surface
        self.interpolation_method = 'cubic' # 'linear', 'cubic', or 'spline'
        self.subpixel_precision = True      # Enable sub-pixel surface detection
        
        # Frame consistency parameters
        self.min_surface_points = 50        # Minimum acceptable surface points
        self.max_surface_points = 5000      # Maximum reasonable surface points
        self.consistency_threshold = 0.3    # Minimum ratio of expected vs actual points
        self.frame_consistency_history = deque(maxlen=10)  # Track recent detection quality
        self.adaptive_threshold_adjustment = True  # Enable adaptive threshold adjustment
        self.last_good_surface = None       # Cache last good surface detection
        self.detection_confidence = 1.0     # Current detection confidence (0-1)

    def select_fluid_color(self, frame):
        """Interactive color selection by clicking on the fluid"""
        self.color_samples = []
        self.color_selected = False
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Sample color from a region around the clicked point
                h, w = frame.shape[:2]
                
                # Define sampling region
                x1 = max(0, x - self.sample_region_size // 2)
                y1 = max(0, y - self.sample_region_size // 2)
                x2 = min(w, x + self.sample_region_size // 2)
                y2 = min(h, y + self.sample_region_size // 2)
                
                # Sample colors from the region
                region = frame[y1:y2, x1:x2]
                region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                region_bgr = region.reshape(-1, 3)
                region_hsv_flat = region_hsv.reshape(-1, 3)
                
                # Store samples
                for bgr_pixel, hsv_pixel in zip(region_bgr, region_hsv_flat):
                    self.color_samples.append({
                        'bgr': bgr_pixel,
                        'hsv': hsv_pixel,
                        'click_pos': (x, y)
                    })
                
                print(f"Color sampled at ({x}, {y})")
                print(f"Sample size: {len(region_bgr)} pixels")
                
                # Visualize sampled region
                temp_frame = frame.copy()
                cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(temp_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('Color Selection', temp_frame)
        
        cv2.namedWindow('Color Selection')
        cv2.setMouseCallback('Color Selection', mouse_callback)
        
        print("\n=== FLUID COLOR SELECTION ===")
        print("Instructions:")
        print("1. Click on the fluid to sample its color")
        print("2. Click multiple times on different fluid areas for better sampling")
        print("3. Press 'c' to confirm selection and calculate color ranges")
        print("4. Press 'r' to reset samples")
        print("5. Press 'q' to quit")
        
        while True:
            temp_frame = frame.copy()
            
            # Show all sampled points
            for i, sample in enumerate(self.color_samples):
                pos = sample['click_pos']
                cv2.circle(temp_frame, pos, 5, (0, 255, 0), -1)
                cv2.putText(temp_frame, str(i+1), (pos[0]+10, pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show sampling info
            cv2.putText(temp_frame, f'Samples: {len(self.color_samples)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if len(self.color_samples) > 0:
                cv2.putText(temp_frame, 'Press C to confirm', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Color Selection', temp_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # Confirm
                if len(self.color_samples) > 0:
                    self.calculate_adaptive_color_ranges()
                    self.color_selected = True
                    break
                else:
                    print("No color samples collected. Please click on the fluid first.")
            elif key == ord('r'):  # Reset
                self.color_samples = []
                print("Color samples reset")
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return True
    
    def calculate_adaptive_color_ranges(self):
        """Calculate adaptive HSV ranges based on color samples"""
        if not self.color_samples:
            print("No color samples to analyze")
            return
        
        # Extract HSV values from samples
        hsv_values = np.array([sample['hsv'] for sample in self.color_samples])
        bgr_values = np.array([sample['bgr'] for sample in self.color_samples])
        
        # Calculate statistics
        hsv_mean = np.mean(hsv_values, axis=0)
        hsv_std = np.std(hsv_values, axis=0)
        bgr_mean = np.mean(bgr_values, axis=0)
        
        # Store target colors
        self.target_color_hsv = hsv_mean.astype(np.uint8)
        self.target_color_bgr = bgr_mean.astype(np.uint8)
        
        # Calculate adaptive ranges with tolerance
        hue_range = max(self.hue_tolerance, hsv_std[0] * 2)
        sat_range = max(self.saturation_tolerance, hsv_std[1] * 2)
        val_range = max(self.value_tolerance, hsv_std[2] * 2)
        
        # Handle hue wrap-around (0-180 in OpenCV)
        hue_mean = hsv_mean[0]
        hue_lower = max(0, hue_mean - hue_range)
        hue_upper = min(180, hue_mean + hue_range)
        
        # Create adaptive ranges
        self.adaptive_hsv_ranges = {
            'lower': np.array([hue_lower, 
                              max(self.min_saturation, hsv_mean[1] - sat_range),
                              max(self.min_value, hsv_mean[2] - val_range)]),
            'upper': np.array([hue_upper,
                              min(255, hsv_mean[1] + sat_range),
                              min(255, hsv_mean[2] + val_range)])
        }
        
        # Handle hue wrap-around for colors near red (0/180 boundary)
        if hue_mean < hue_range:
            # Color is near 0, create additional range near 180
            self.adaptive_hsv_ranges['lower2'] = np.array([180 - (hue_range - hue_mean),
                                                          max(self.min_saturation, hsv_mean[1] - sat_range),
                                                          max(self.min_value, hsv_mean[2] - val_range)])
            self.adaptive_hsv_ranges['upper2'] = np.array([180,
                                                          min(255, hsv_mean[1] + sat_range),
                                                          min(255, hsv_mean[2] + val_range)])
        elif hue_mean > 180 - hue_range:
            # Color is near 180, create additional range near 0
            self.adaptive_hsv_ranges['lower2'] = np.array([0,
                                                          max(self.min_saturation, hsv_mean[1] - sat_range),
                                                          max(self.min_value, hsv_mean[2] - val_range)])
            self.adaptive_hsv_ranges['upper2'] = np.array([hue_range - (180 - hue_mean),
                                                          min(255, hsv_mean[1] + sat_range),
                                                          min(255, hsv_mean[2] + val_range)])
        
        print(f"\n=== ADAPTIVE COLOR RANGES CALCULATED ===")
        print(f"Samples analyzed: {len(self.color_samples)}")
        print(f"Target HSV: {self.target_color_hsv}")
        print(f"Target BGR: {self.target_color_bgr}")
        print(f"HSV Range: {self.adaptive_hsv_ranges['lower']} - {self.adaptive_hsv_ranges['upper']}")
        if 'lower2' in self.adaptive_hsv_ranges:
            print(f"Additional HSV Range: {self.adaptive_hsv_ranges['lower2']} - {self.adaptive_hsv_ranges['upper2']}")
    
    def save_color_profile(self, video_name):
        """Save current color profile to file"""
        if not self.color_selected:
            print("No color profile to save")
            return
        
        profile_data = {
            'target_color_bgr': self.target_color_bgr.tolist() if self.target_color_bgr is not None else None,
            'target_color_hsv': self.target_color_hsv.tolist() if self.target_color_hsv is not None else None,
            'adaptive_hsv_ranges': {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in self.adaptive_hsv_ranges.items()
            } if self.adaptive_hsv_ranges else None,
            'color_tolerance': self.color_tolerance,
            'hue_tolerance': self.hue_tolerance,
            'saturation_tolerance': self.saturation_tolerance,
            'value_tolerance': self.value_tolerance
        }
        
        # Create filename from video name
        profile_filename = os.path.join(self.color_profiles_dir, f"{video_name}_color_profile.json")
        
        with open(profile_filename, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        print(f"Color profile saved to: {profile_filename}")
    
    def load_color_profile(self, video_name):
        """Load color profile from file"""
        profile_filename = os.path.join(self.color_profiles_dir, f"{video_name}_color_profile.json")
        
        if not os.path.exists(profile_filename):
            return False
        
        try:
            with open(profile_filename, 'r') as f:
                profile_data = json.load(f)
            
            # Restore color data
            self.target_color_bgr = np.array(profile_data['target_color_bgr']) if profile_data['target_color_bgr'] else None
            self.target_color_hsv = np.array(profile_data['target_color_hsv']) if profile_data['target_color_hsv'] else None
            
            if profile_data['adaptive_hsv_ranges']:
                self.adaptive_hsv_ranges = {
                    key: np.array(value) if isinstance(value, list) else value
                    for key, value in profile_data['adaptive_hsv_ranges'].items()
                }
            
            # Restore tolerance settings
            self.color_tolerance = profile_data.get('color_tolerance', self.color_tolerance)
            self.hue_tolerance = profile_data.get('hue_tolerance', self.hue_tolerance)
            self.saturation_tolerance = profile_data.get('saturation_tolerance', self.saturation_tolerance)
            self.value_tolerance = profile_data.get('value_tolerance', self.value_tolerance)
            
            self.color_selected = True
            print(f"Color profile loaded from: {profile_filename}")
            print(f"Target HSV: {self.target_color_hsv}")
            return True
            
        except Exception as e:
            print(f"Error loading color profile: {e}")
            return False
    
    class SimpleKalmanFilter:
        """Simple 1D Kalman filter for surface point tracking - improved for fast movements"""
        def __init__(self, initial_value, process_noise=2.0, measurement_noise=8.0, fast_motion_mode=False):
            self.x = initial_value  # state estimate
            self.P = 1.0           # error covariance
            self.Q = process_noise  # process noise
            self.R = measurement_noise  # measurement noise
            self.K = 0.0           # Kalman gain
            self.fast_motion_mode = fast_motion_mode
            
            # For fast motion, increase process noise and reduce measurement noise weight
            if fast_motion_mode:
                self.Q = process_noise * 2.0  # Higher process noise for fast motion
                self.R = measurement_noise * 0.7  # Lower measurement noise for faster response
            
        def predict(self):
            """Predict step"""
            # For simplicity, assume constant position model
            # x_k = x_{k-1} (no motion model)
            self.P = self.P + self.Q
            return self.x
            
        def update(self, measurement):
            """Update step"""
            # Kalman gain
            self.K = self.P / (self.P + self.R)
            
            # Update estimate
            self.x = self.x + self.K * (measurement - self.x)
            
            # Update error covariance
            self.P = (1 - self.K) * self.P
            
            return self.x
        
        def set_fast_motion_mode(self, fast_motion):
            """Adjust parameters for fast motion"""
            if fast_motion != self.fast_motion_mode:
                self.fast_motion_mode = fast_motion
                if fast_motion:
                    self.Q = self.Q * 2.0 if self.Q < 4.0 else self.Q
                    self.R = self.R * 0.7 if self.R > 2.0 else self.R
                else:
                    self.Q = max(1.0, self.Q * 0.5)
                    self.R = min(10.0, self.R * 1.4)
    
    def ransac_line_fitting(self, points, max_iterations=100, threshold=10.0, min_samples=10):
        """
        RANSAC-based line fitting to remove outliers from surface points
        
        Args:
            points: List of (x, y) tuples representing surface points
            max_iterations: Maximum number of RANSAC iterations
            threshold: Maximum distance for a point to be considered an inlier
            min_samples: Minimum number of samples to estimate a line
            
        Returns:
            best_inliers: List of inlier points
            best_line_params: (slope, intercept) of the best line
        """
        if len(points) < min_samples:
            return points, None
            
        points = np.array(points)
        best_inliers = []
        best_line_params = None
        best_score = 0
        
        for _ in range(max_iterations):
            # Randomly sample min_samples points
            sample_indices = random.sample(range(len(points)), min_samples)
            sample_points = points[sample_indices]
            
            # Fit line to sample points using least squares
            try:
                A = np.vstack([sample_points[:, 0], np.ones(len(sample_points))]).T
                line_params = np.linalg.lstsq(A, sample_points[:, 1], rcond=None)[0]
                slope, intercept = line_params
            except:
                continue
            
            # Find inliers
            inliers = []
            for i, point in enumerate(points):
                x, y = point
                predicted_y = slope * x + intercept
                distance = abs(y - predicted_y)
                
                if distance < threshold:
                    inliers.append(i)
            
            # Check if this is the best model so far
            if len(inliers) > best_score:
                best_score = len(inliers)
                best_inliers = [points[i] for i in inliers]
                best_line_params = (slope, intercept)
        
        # Refine the best line using all inliers
        if best_inliers and len(best_inliers) >= min_samples:
            inlier_points = np.array(best_inliers)
            try:
                A = np.vstack([inlier_points[:, 0], np.ones(len(inlier_points))]).T
                refined_params = np.linalg.lstsq(A, inlier_points[:, 1], rcond=None)[0]
                best_line_params = tuple(refined_params)
            except:
                pass
        
        return best_inliers, best_line_params
    
    def polynomial_ransac_fitting(self, points, degree=2, max_iterations=100, threshold=15.0):
        """
        RANSAC-based polynomial fitting for curved surfaces
        
        Args:
            points: List of (x, y) tuples representing surface points
            degree: Degree of polynomial (2 for quadratic, 3 for cubic)
            max_iterations: Maximum number of RANSAC iterations
            threshold: Maximum distance for a point to be considered an inlier
            
        Returns:
            best_inliers: List of inlier points
            best_poly_params: Polynomial coefficients
        """
        if len(points) < degree + 1:
            return points, None
            
        points = np.array(points)
        best_inliers = []
        best_poly_params = None
        best_score = 0
        min_samples = degree + 1
        
        for _ in range(max_iterations):
            # Randomly sample points
            if len(points) < min_samples:
                break
                
            sample_indices = random.sample(range(len(points)), min_samples)
            sample_points = points[sample_indices]
            
            # Fit polynomial to sample points
            try:
                poly_params = np.polyfit(sample_points[:, 0], sample_points[:, 1], degree)
            except:
                continue
            
            # Find inliers
            inliers = []
            for i, point in enumerate(points):
                x, y = point
                predicted_y = np.polyval(poly_params, x)
                distance = abs(y - predicted_y)
                
                if distance < threshold:
                    inliers.append(i)
            
            # Check if this is the best model so far
            if len(inliers) > best_score:
                best_score = len(inliers)
                best_inliers = [points[i] for i in inliers]
                best_poly_params = poly_params
        
        # Refine the best polynomial using all inliers
        if best_inliers and len(best_inliers) >= min_samples:
            inlier_points = np.array(best_inliers)
            try:
                refined_params = np.polyfit(inlier_points[:, 0], inlier_points[:, 1], degree)
                best_poly_params = refined_params
            except:
                pass
        
        return best_inliers, best_poly_params
    
    def apply_physics_constraints(self, points):
        """
        Apply physics-based constraints to surface points
        
        Args:
            points: List of (x, y) tuples representing surface points
            
        Returns:
            filtered_points: Points that satisfy physics constraints
        """
        if len(points) < 3:
            return points
            
        points = sorted(points, key=lambda p: p[0])  # Sort by x-coordinate
        filtered_points = []
        
        for i, (x, y) in enumerate(points):
            valid = True
            
            # Check slope constraint with neighboring points
            if i > 0:
                prev_x, prev_y = points[i-1]
                slope = (y - prev_y) / (x - prev_x) if x != prev_x else 0
                if abs(slope) > self.max_surface_slope:
                    valid = False
            
            if i < len(points) - 1:
                next_x, next_y = points[i+1]
                slope = (next_y - y) / (next_x - x) if next_x != x else 0
                if abs(slope) > self.max_surface_slope:
                    valid = False
            
            # Check smoothness constraint (curvature)
            if i > 0 and i < len(points) - 1:
                prev_x, prev_y = points[i-1]
                next_x, next_y = points[i+1]
                
                # Calculate second derivative (curvature approximation)
                if next_x != prev_x:
                    curvature = 2 * (y - (prev_y + next_y) / 2) / ((next_x - prev_x) ** 2)
                    if abs(curvature) > 0.01:  # Maximum curvature constraint
                        valid = False
            
            if valid:
                filtered_points.append((x, y))
        
        return filtered_points
    
    def temporal_smoothing(self, current_points):
        """
        Apply temporal smoothing using Kalman filters with motion-adaptive parameters
        
        Args:
            current_points: List of (x, y) tuples from current frame
            
        Returns:
            smoothed_points: Temporally smoothed surface points
        """
        if not self.use_temporal_smoothing or not current_points:
            return current_points
        
        # Estimate motion speed and adapt parameters
        self.estimate_motion_speed(current_points)
        
        smoothed_points = []
        is_fast_motion = self.motion_speed > self.fast_motion_threshold
        
        # Convert to dictionary for easier lookup
        current_dict = {x: y for x, y in current_points}
        
        # Update or create Kalman filters for each x-coordinate
        for x, y in current_points:
            if x not in self.kalman_filters:
                # Create new Kalman filter with motion-adaptive parameters
                self.kalman_filters[x] = self.SimpleKalmanFilter(
                    initial_value=y,
                    process_noise=2.0,
                    measurement_noise=8.0,
                    fast_motion_mode=is_fast_motion
                )
            
            # Update filter
            kalman_filter = self.kalman_filters[x]
            predicted_y = kalman_filter.predict()
            smoothed_y = kalman_filter.update(y)
            
            # Check for temporal consistency with adaptive threshold
            if (self.prev_surface_points and 
                any(abs(x - prev_x) < 5 for prev_x, _ in self.prev_surface_points)):
                # Find closest previous point
                closest_prev = min(self.prev_surface_points, 
                                 key=lambda p: abs(p[0] - x))
                prev_y = closest_prev[1]
                
                # Use adaptive temporal jump threshold
                current_max_jump = self.max_temporal_jump
                if is_fast_motion:
                    current_max_jump = min(150, self.max_temporal_jump + self.motion_speed)
                
                # Reject if jump is too large
                if abs(smoothed_y - prev_y) > current_max_jump:
                    smoothed_y = prev_y + np.sign(smoothed_y - prev_y) * current_max_jump
            
            smoothed_points.append((x, int(smoothed_y)))
        
        # Clean up old filters
        current_x_coords = set(x for x, _ in current_points)
        keys_to_remove = [x for x in self.kalman_filters.keys() if x not in current_x_coords]
        for x in keys_to_remove:
            del self.kalman_filters[x]
        
        self.prev_surface_points = smoothed_points
        return smoothed_points
    
    def estimate_motion_speed(self, current_points):
        """Estimate current motion speed to adapt parameters"""
        if not self.prev_surface_points or not current_points:
            self.motion_speed = 0.0
            return
        
        # Calculate average displacement between frames
        total_displacement = 0
        matching_points = 0
        
        for curr_x, curr_y in current_points:
            # Find closest point in previous frame
            min_distance = float('inf')
            closest_prev_y = None
            
            for prev_x, prev_y in self.prev_surface_points:
                if abs(curr_x - prev_x) < 10:  # Only consider nearby x-coordinates
                    distance = abs(curr_y - prev_y)
                    if distance < min_distance:
                        min_distance = distance
                        closest_prev_y = prev_y
            
            if closest_prev_y is not None:
                total_displacement += abs(curr_y - closest_prev_y)
                matching_points += 1
        
        # Update motion speed estimate
        if matching_points > 0:
            frame_displacement = total_displacement / matching_points
            # Smooth the motion speed estimate
            self.motion_speed = 0.7 * self.motion_speed + 0.3 * frame_displacement
        
        # Adapt parameters based on motion speed
        self.adapt_parameters_for_motion()
    
    def adapt_parameters_for_motion(self):
        """Adapt detection parameters based on current motion speed"""
        is_fast_motion = self.motion_speed > self.fast_motion_threshold
        
        if is_fast_motion:
            # Increase temporal tolerance for fast motion
            if self.adaptive_temporal_jump:
                self.max_temporal_jump = min(100, 60 + self.motion_speed)
            
            # Adjust Kalman filters for fast motion
            for x_coord in self.kalman_filters:
                self.kalman_filters[x_coord].set_fast_motion_mode(True)
                
            # Relax RANSAC parameters for fast motion
            self.ransac_threshold = min(25.0, 15.0 + self.motion_speed * 0.2)
            self.min_inlier_ratio = max(0.15, 0.25 - self.motion_speed * 0.005)
        else:
            # Reset to normal parameters for slow motion
            if self.adaptive_temporal_jump:
                self.max_temporal_jump = 60
            
            # Adjust Kalman filters for normal motion
            for x_coord in self.kalman_filters:
                self.kalman_filters[x_coord].set_fast_motion_mode(False)
                
            # Reset RANSAC parameters
            self.ransac_threshold = 15.0
            self.min_inlier_ratio = 0.25
            
        # Adapt color detection for motion blur
        self.adapt_color_detection_for_motion()
    
    def adapt_color_detection_for_motion(self):
        """Adapt color detection parameters based on motion speed to handle motion blur"""
        is_fast_motion = self.motion_speed > self.fast_motion_threshold
        
        if is_fast_motion:
            # Increase tolerance for motion blur
            self.color_tolerance = min(80, 40 + self.motion_speed * 0.5)
            self.hue_tolerance = min(40, 20 + self.motion_speed * 0.3)
            self.saturation_tolerance = min(100, 60 + self.motion_speed * 0.5)
            self.value_tolerance = min(120, 80 + self.motion_speed * 0.4)
        else:
            # Reset to normal tolerance
            self.color_tolerance = 40
            self.hue_tolerance = 20
            self.saturation_tolerance = 60
            self.value_tolerance = 80
    
    def detect_colored_fluid(self, frame):
        """Detect fluid using adaptive color ranges or fallback to default"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        
        if self.color_selected and self.adaptive_hsv_ranges:
            # Use adaptive color detection
            mask = cv2.inRange(hsv, self.adaptive_hsv_ranges['lower'], self.adaptive_hsv_ranges['upper'])
            
            # Handle hue wrap-around if needed
            if 'lower2' in self.adaptive_hsv_ranges:
                mask2 = cv2.inRange(hsv, self.adaptive_hsv_ranges['lower2'], self.adaptive_hsv_ranges['upper2'])
                mask = cv2.bitwise_or(mask, mask2)
        else:
            # Use default pink color detection
            mask_pink1 = cv2.inRange(hsv, self.lower_pink1, self.upper_pink1)
            mask_pink2 = cv2.inRange(hsv, self.lower_pink2, self.upper_pink2)
            mask = cv2.bitwise_or(mask_pink1, mask_pink2)
        
        # Additional filtering to exclude container edges
        # Create a mask to exclude image borders (likely container edges)
        border_mask = np.ones_like(mask)
        border_mask[:self.edge_margin, :] = 0  # Top edge
        border_mask[-self.edge_margin:, :] = 0  # Bottom edge
        border_mask[:, :self.edge_margin] = 0   # Left edge
        border_mask[:, -self.edge_margin:] = 0  # Right edge
        
        mask = cv2.bitwise_and(mask, border_mask)
        
        # Apply adaptive morphological operations based on motion speed and detection confidence
        kernel_size = self.kernel_size
        if self.motion_speed > self.fast_motion_threshold:
            # Use larger kernel for fast motion to handle motion blur
            kernel_size = min(7, self.kernel_size + int(self.motion_speed * 0.1))
        
        # Adjust kernel size based on detection confidence
        if hasattr(self, 'detection_confidence') and self.detection_confidence < 0.6:
            kernel_size = min(9, kernel_size + 2)  # Larger kernel for poor detection
        
        adaptive_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply different morphological operations based on detection quality
        if hasattr(self, 'detection_confidence') and self.detection_confidence < 0.5:
            # More aggressive morphological operations for poor detection
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, adaptive_kernel)  # Fill gaps first
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, adaptive_kernel)   # Then remove noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, adaptive_kernel)  # Fill gaps again
        else:
            # Standard morphological operations
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, adaptive_kernel)  # Remove noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, adaptive_kernel)  # Fill gaps
        
        # Apply additional filtering based on color similarity if we have adaptive colors
        if self.color_selected and self.target_color_bgr is not None:
            refined_mask = np.zeros_like(mask)
            
            # Convert frame to BGR for color distance calculation
            frame_bgr = frame
            
            # Only process areas where the initial mask is positive
            mask_indices = np.where(mask > 0)
            
            if len(mask_indices[0]) > 0:
                # Calculate color distances for pixels in the mask
                pixels = frame_bgr[mask_indices]
                distances = np.sqrt(np.sum((pixels - self.target_color_bgr)**2, axis=1))
                
                # Keep only pixels within color tolerance
                valid_pixels = distances < self.color_tolerance
                refined_mask[mask_indices[0][valid_pixels], mask_indices[1][valid_pixels]] = 255
            
            # Apply final smoothing
            refined_mask = cv2.GaussianBlur(refined_mask, (3, 3), 0)
            return refined_mask
        else:
            # Apply basic smoothing
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            return mask
    
    def find_fluid_surface_line(self, frame):
        """Find the surface line of the colored fluid with improved filtering and consistency checking"""
        h, w = frame.shape[:2]
        
        # Get colored fluid mask
        fluid_mask = self.detect_colored_fluid(frame)
        
        # Apply ROI if manually selected
        if self.roi_selected and self.selected_roi:
            x, y, w_roi, h_roi = self.selected_roi
            roi_mask = np.zeros_like(fluid_mask)
            roi_mask[y:y+h_roi, x:x+w_roi] = 255
            fluid_mask = cv2.bitwise_and(fluid_mask, roi_mask)
        else:
            # Apply automatic ROI
            roi_y1 = int(h * self.roi_top)
            roi_y2 = int(h * self.roi_bottom)
            roi_mask = np.zeros_like(fluid_mask)
            roi_mask[roi_y1:roi_y2, :] = 255
            fluid_mask = cv2.bitwise_and(fluid_mask, roi_mask)
        
        # Try multiple detection methods with increasing aggressiveness
        surface_points = self.robust_surface_detection_with_fallbacks(fluid_mask, w)
        
        # Check frame consistency and apply corrections if needed
        surface_points = self.ensure_frame_consistency(surface_points, frame)
        
        return surface_points, fluid_mask
    
    def robust_surface_detection_with_fallbacks(self, fluid_mask, width):
        """Try multiple detection methods with increasing aggressiveness"""
        
        # Method 1: Standard contour-based detection
        surface_points = self.extract_surface_from_contours(fluid_mask, width)
        
        if self.is_detection_reasonable(surface_points):
            return surface_points
        
        # Method 2: Relaxed contour detection
        original_min_area = self.min_contour_area
        original_edge_margin = self.edge_margin
        
        try:
            # Relax parameters for better detection
            self.min_contour_area = max(100, self.min_contour_area * 0.5)
            self.edge_margin = max(5, self.edge_margin * 0.5)
            
            surface_points = self.extract_surface_from_contours(fluid_mask, width)
            
            if self.is_detection_reasonable(surface_points):
                return surface_points
                
        finally:
            # Restore original parameters
            self.min_contour_area = original_min_area
            self.edge_margin = original_edge_margin
        
        # Method 3: Mask-based detection
        surface_points = self.extract_surface_from_mask(fluid_mask, width)
        
        if self.is_detection_reasonable(surface_points):
            return surface_points
        
        # Method 4: Aggressive mask-based detection
        original_density = self.surface_point_density
        try:
            # Increase density for better coverage
            self.surface_point_density = min(4, self.surface_point_density * 1.5)
            surface_points = self.extract_surface_from_mask(fluid_mask, width)
            
            if self.is_detection_reasonable(surface_points):
                return surface_points
                
        finally:
            self.surface_point_density = original_density
        
        # Method 5: Enhanced mask with morphological operations
        surface_points = self.extract_surface_with_enhanced_morphology(fluid_mask, width)
        
        return surface_points if surface_points else []
    
    def is_detection_reasonable(self, surface_points):
        """Check if surface detection results are reasonable"""
        if not surface_points:
            return False
            
        num_points = len(surface_points)
        
        # Basic count check
        if num_points < self.min_surface_points or num_points > self.max_surface_points:
            return False
        
        # Check if points form a reasonable surface line
        if num_points > 5:
            x_coords = [p[0] for p in surface_points]
            y_coords = [p[1] for p in surface_points]
            
            # Check x-coordinate distribution
            x_range = max(x_coords) - min(x_coords)
            if x_range < 100:  # Surface should span reasonable width
                return False
            
            # Check y-coordinate variation (shouldn't be too erratic)
            y_std = np.std(y_coords)
            if y_std > 100:  # Too much variation suggests poor detection
                return False
        
        return True
    
    def ensure_frame_consistency(self, surface_points, frame):
        """Ensure detection consistency across frames"""
        current_quality = self.calculate_detection_quality(surface_points)
        self.frame_consistency_history.append(current_quality)
        
        # Update detection confidence
        if len(self.frame_consistency_history) > 3:
            recent_avg = np.mean(list(self.frame_consistency_history)[-3:])
            self.detection_confidence = max(0.1, min(1.0, recent_avg))
        
        # If current detection is poor but we have good history, try to recover
        if (current_quality < 0.5 and 
            len(self.frame_consistency_history) > 2 and
            np.mean(list(self.frame_consistency_history)[-3:-1]) > 0.7):
            
            # Try temporal consistency recovery
            recovered_points = self.recover_using_temporal_consistency(surface_points, frame)
            if self.is_detection_reasonable(recovered_points):
                return recovered_points
        
        # Cache good detections for future use
        if current_quality > 0.8:
            self.last_good_surface = surface_points.copy()
        
        return surface_points
    
    def calculate_detection_quality(self, surface_points):
        """Calculate quality score for surface detection (0-1)"""
        if not surface_points:
            return 0.0
        
        num_points = len(surface_points)
        
        # Base score from point count
        optimal_count = 500  # Reasonable number of surface points
        count_score = min(1.0, num_points / optimal_count)
        
        if num_points < self.min_surface_points:
            count_score *= 0.3
        elif num_points > self.max_surface_points:
            count_score *= 0.6
        
        # Continuity score
        continuity_score = 1.0
        if num_points > 5:
            x_coords = sorted([p[0] for p in surface_points])
            gaps = []
            for i in range(1, len(x_coords)):
                gap = x_coords[i] - x_coords[i-1]
                gaps.append(gap)
            
            if gaps:
                avg_gap = np.mean(gaps)
                max_gap = max(gaps)
                if max_gap > avg_gap * 3:  # Large gaps indicate poor continuity
                    continuity_score *= 0.7
        
        # Smoothness score
        smoothness_score = 1.0
        if num_points > 10:
            y_coords = [p[1] for p in surface_points]
            y_std = np.std(y_coords)
            if y_std > 50:  # Too much variation
                smoothness_score *= 0.8
        
        # Combined score
        quality = (count_score * 0.5 + continuity_score * 0.3 + smoothness_score * 0.2)
        return max(0.0, min(1.0, quality))
    
    def recover_using_temporal_consistency(self, poor_points, frame):
        """Try to recover poor detection using temporal information"""
        if not self.last_good_surface:
            return poor_points
        
        # Use last good surface as a template
        template_points = self.last_good_surface
        
        # Try to find similar points in current frame
        h, w = frame.shape[:2]
        recovered_points = []
        
        # Get fluid mask for reference
        fluid_mask = self.detect_colored_fluid(frame)
        
        # For each point in the template, try to find corresponding point in current frame
        for template_x, template_y in template_points[::5]:  # Sample every 5th point
            # Search in a neighborhood around template position
            search_radius = 30
            best_y = template_y
            best_score = 0
            
            for dy in range(-search_radius, search_radius + 1):
                y_candidate = template_y + dy
                if 0 <= y_candidate < h and 0 <= template_x < w:
                    # Check if this position has fluid
                    fluid_score = fluid_mask[y_candidate, template_x] / 255.0
                    
                    # Prefer positions closer to template
                    distance_score = 1.0 - abs(dy) / search_radius
                    
                    total_score = fluid_score * 0.7 + distance_score * 0.3
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_y = y_candidate
            
            if best_score > 0.3:  # Minimum confidence threshold
                recovered_points.append((template_x, best_y))
        
        # Interpolate between recovered points for smooth surface
        if len(recovered_points) > 10:
            return self.interpolate_surface_line(recovered_points, w)
        
        return poor_points
    
    def extract_surface_with_enhanced_morphology(self, fluid_mask, width):
        """Extract surface using enhanced morphological operations"""
        # Apply stronger morphological operations
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Close gaps more aggressively
        enhanced_mask = cv2.morphologyEx(fluid_mask, cv2.MORPH_CLOSE, kernel_large)
        
        # Fill holes
        enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_OPEN, kernel_large)
        
        # Try edge detection on enhanced mask
        edges = cv2.Canny(enhanced_mask, 50, 150)
        
        # Find contours on edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        surface_points = []
        if contours:
            # Use the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                surface_points = self.extract_surface_from_contours(enhanced_mask, width)
        
        # Fallback to mask-based method
        if not surface_points:
            surface_points = self.extract_surface_from_mask(enhanced_mask, width)
        
        return surface_points
    
    def force_detection_recovery(self):
        """Force reset detection parameters and clear cache for recovery"""
        # Reset detection confidence
        self.detection_confidence = 1.0
        
        # Clear frame consistency history
        self.frame_consistency_history.clear()
        
        # Clear cached surface
        self.last_good_surface = None
        
        # Reset to conservative detection parameters
        self.min_contour_area = 500
        self.edge_margin = 20
        self.min_surface_points = 50
        self.max_surface_points = 5000
        
        # Reset color tolerances to default
        self.color_tolerance = 40
        self.hue_tolerance = 20
        self.saturation_tolerance = 60
        self.value_tolerance = 80
        
        # Reset RANSAC parameters
        self.ransac_threshold = 15.0
        self.min_inlier_ratio = 0.25
        
        # Clear Kalman filters
        self.kalman_filters.clear()
        self.prev_surface_points = None
        
        # Reset motion tracking
        self.motion_speed = 0.0
        
        print("Detection system reset to default parameters")
    
    def check_automatic_recovery(self):
        """Check if automatic recovery should be triggered based on detection history"""
        if len(self.frame_consistency_history) < 8:
            return
        
        # Check if detection quality has been consistently poor
        recent_quality = list(self.frame_consistency_history)[-8:]
        avg_quality = np.mean(recent_quality)
        
        # Trigger recovery if quality is consistently poor
        if avg_quality < 0.4:
            print(f"Automatic recovery triggered - avg quality: {avg_quality:.2f}")
            self.force_detection_recovery()
        
        # Also check for sudden drops in detection quality
        if len(self.frame_consistency_history) >= 3:
            recent_3 = list(self.frame_consistency_history)[-3:]
            if all(q < 0.3 for q in recent_3):
                print("Sudden detection failure detected - triggering recovery")
                self.force_detection_recovery()
    
    def extract_surface_from_contours(self, fluid_mask, width):
        """Extract surface using contour-based method"""
        # Find contours in the fluid mask
        contours, _ = cv2.findContours(fluid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        surface_points = []
        if contours:
            # Filter contours to find the main fluid body
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_contour_area:
                    # Check if contour is not just at the edges (likely container)
                    x, y, w_cont, h_cont = cv2.boundingRect(contour)
                    
                    # Skip contours that are too close to image edges
                    if (x > self.edge_margin and 
                        x + w_cont < width - self.edge_margin and
                        y > self.edge_margin and 
                        w_cont > self.min_fluid_width):
                        valid_contours.append(contour)
            
            if valid_contours:
                # Use the largest valid contour (main fluid body)
                largest_contour = max(valid_contours, key=cv2.contourArea)
                
                # Extract surface line from the main fluid contour
                surface_points = self.extract_surface_line(largest_contour, width)
                
                # Additional validation: check if surface line looks reasonable
                if surface_points:
                    # Ensure surface line is not too fragmented
                    x_coords = [p[0] for p in surface_points]
                    if len(x_coords) > 1:
                        x_range = max(x_coords) - min(x_coords)
                        # If the surface line is too fragmented, reject it
                        if x_range < self.min_fluid_width:
                            surface_points = []
        
        return surface_points
    
    def extract_surface_from_mask(self, fluid_mask, width):
        """Fallback method: Extract surface directly from fluid mask with high density"""
        h, w = fluid_mask.shape
        surface_points = []
        
        # For each column, find the topmost fluid pixel with sub-pixel precision
        step_size = 1.0 / self.surface_point_density  # Generate more points between pixels
        
        for x_float in np.arange(self.edge_margin, w - self.edge_margin, step_size):
            x = int(x_float)
            
            # Find the topmost fluid pixel in this column
            for y in range(self.edge_margin, h - self.edge_margin):
                if fluid_mask[y, x] > 0:  # Found fluid pixel
                    # Add sub-pixel precision if enabled
                    if self.subpixel_precision and x > 0 and x < w - 1:
                        # Simple sub-pixel interpolation
                        y_subpixel = self.subpixel_surface_detection(fluid_mask, x, y)
                        surface_points.append((x_float, y_subpixel))
                    else:
                        surface_points.append((x_float, float(y)))
                    break  # Take only the topmost pixel
        
        # Filter out isolated points
        if len(surface_points) < 10:  # Need minimum number of points
            return []
        
        # Remove isolated points by checking neighbors
        filtered_points = []
        for i, (x, y) in enumerate(surface_points):
            # Check if this point has neighbors within reasonable distance
            has_neighbors = False
            for j, (x2, y2) in enumerate(surface_points):
                if i != j and abs(x - x2) < 15 and abs(y - y2) < 20:
                    has_neighbors = True
                    break
            
            if has_neighbors or len(surface_points) < 20:
                filtered_points.append((x, y))
        
        # Apply advanced smoothing
        if len(filtered_points) > 5:
            filtered_points = self.advanced_surface_smoothing(filtered_points)
        
        # Convert back to integer coordinates for compatibility
        return [(int(x), int(y)) for x, y in filtered_points]
    
    def subpixel_surface_detection(self, fluid_mask, x, y):
        """Detect surface position with sub-pixel precision using gradient analysis"""
        h, w = fluid_mask.shape
        
        # Advanced sub-pixel detection using gradient analysis
        if y > 1 and y < h - 2 and x > 1 and x < w - 2:
            # Calculate gradients in a 3x3 neighborhood
            gradient_y = 0.0
            gradient_weights = 0.0
            
            # Use Sobel-like gradient calculation
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dy != 0:  # Only consider vertical gradient
                        nx, ny = x + dx, y + dy
                        weight = 2.0 if dx == 0 else 1.0  # Center column has more weight
                        gradient_y += weight * float(fluid_mask[ny, nx]) * dy
                        gradient_weights += weight
            
            if gradient_weights > 0:
                gradient_y /= gradient_weights
                
                # Find the zero-crossing of the gradient (edge position)
                if abs(gradient_y) > 10:  # Minimum gradient threshold
                    # Check gradients at adjacent positions
                    prev_gradient = 0.0
                    next_gradient = 0.0
                    
                    if y > 0:
                        prev_gradient = float(fluid_mask[y-1, x]) - float(fluid_mask[y, x])
                    if y < h - 1:
                        next_gradient = float(fluid_mask[y, x]) - float(fluid_mask[y+1, x])
                    
                    # Interpolate to find sub-pixel position
                    if prev_gradient != 0 and next_gradient != 0:
                        # Use quadratic interpolation for better precision
                        denominator = prev_gradient - 2*gradient_y + next_gradient
                        if abs(denominator) > 1e-6:
                            adjustment = 0.5 * (prev_gradient - next_gradient) / denominator
                            # Clamp adjustment to reasonable range
                            adjustment = max(-0.5, min(0.5, adjustment))
                            return max(0, y + adjustment)
        
        return float(y)
    
    def advanced_surface_smoothing(self, surface_points):
        """Apply advanced smoothing to surface points"""
        if len(surface_points) < 5:
            return surface_points
        
        # Sort by x-coordinate
        surface_points.sort(key=lambda p: p[0])
        
        # Extract coordinates
        x_coords = [p[0] for p in surface_points]
        y_coords = [p[1] for p in surface_points]
        
        # Apply smoothing based on selected method
        if self.interpolation_method == 'cubic' and len(surface_points) > 10:
            # Cubic spline smoothing
            try:
                from scipy.interpolate import CubicSpline
                cs = CubicSpline(x_coords, y_coords)
                
                # Generate smooth curve
                x_smooth = np.linspace(min(x_coords), max(x_coords), 
                                     int((max(x_coords) - min(x_coords)) * self.surface_point_density))
                y_smooth = cs(x_smooth)
                
                return list(zip(x_smooth, y_smooth))
            except:
                # Fall back to linear interpolation
                pass
        
        # Enhanced median filtering
        smoothed_points = []
        window_size = 7  # Increased window size
        
        for i in range(len(surface_points)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(surface_points), i + window_size // 2 + 1)
            
            window_points = surface_points[start_idx:end_idx]
            
            # Use weighted median (give more weight to center points)
            weights = np.exp(-np.abs(np.arange(len(window_points)) - len(window_points)//2))
            y_values = [p[1] for p in window_points]
            
            # Weighted average instead of median for smoother results
            avg_y = np.average(y_values, weights=weights)
            smoothed_points.append((surface_points[i][0], avg_y))
        
        return smoothed_points
    
    def robust_surface_extraction(self, contour, width):
        """
        Robust surface line extraction using RANSAC + Kalman filtering + Physics constraints
        
        Args:
            contour: OpenCV contour of the fluid
            width: Frame width for boundary checking
            
        Returns:
            surface_points: List of (x, y) tuples representing the robust surface line
        """
        # Step 1: Extract initial surface points from contour
        initial_points = self.extract_initial_surface_points(contour, width)
        
        if len(initial_points) < self.ransac_min_samples:
            return []
        
        # Step 2: Apply RANSAC to remove outliers
        if self.use_robust_detection:
            # Try polynomial fitting first for curved surfaces
            inlier_points, poly_params = self.polynomial_ransac_fitting(
                initial_points, 
                degree=2, 
                max_iterations=self.ransac_iterations,
                threshold=self.ransac_threshold
            )
            
            # If polynomial fitting doesn't work well, fall back to linear RANSAC
            if not inlier_points or len(inlier_points) < len(initial_points) * self.min_inlier_ratio:
                inlier_points, line_params = self.ransac_line_fitting(
                    initial_points,
                    max_iterations=self.ransac_iterations,
                    threshold=self.ransac_threshold,
                    min_samples=self.ransac_min_samples
                )
            
            # If RANSAC fails, use original points
            if not inlier_points:
                inlier_points = initial_points
        else:
            inlier_points = initial_points
        
        # Step 3: Apply physics-based constraints
        physics_filtered_points = self.apply_physics_constraints(inlier_points)
        
        # Step 4: Apply temporal smoothing with Kalman filtering
        smoothed_points = self.temporal_smoothing(physics_filtered_points)
        
        # Step 5: Interpolate to create continuous surface line
        if len(smoothed_points) > 1:
            final_points = self.interpolate_surface_line(smoothed_points, width)
        else:
            final_points = smoothed_points
        
        return final_points
    
    def extract_initial_surface_points(self, contour, width):
        """Extract initial surface points from contour - improved for high-density surface detection"""
        # Get all points from the contour
        contour_points = contour.reshape(-1, 2)
        
        # Filter out points too close to image edges (likely container edges)
        filtered_points = []
        for point in contour_points:
            x, y = point
            if (self.edge_margin < x < width - self.edge_margin and 
                y > self.edge_margin):  # Allow points near top but not other edges
                filtered_points.append(tuple(point))
        
        if len(filtered_points) < 3:  # Need at least 3 points for a line
            return []
        
        # NEW APPROACH: Find the actual top surface using contour analysis
        # Step 1: Find the topmost point overall to establish surface direction
        min_y = min(point[1] for point in filtered_points)
        top_threshold = min_y + 20  # Allow some tolerance for surface thickness
        
        # Step 2: Get all points that are near the top surface
        surface_candidates = []
        for x, y in filtered_points:
            if y <= top_threshold:  # Only consider points near the top
                surface_candidates.append((x, y))
        
        if len(surface_candidates) < 3:
            return []
        
        # Step 3: Create high-density surface points using interpolation
        # Group points by x-coordinate ranges for better surface detection
        x_min = min(point[0] for point in surface_candidates)
        x_max = max(point[0] for point in surface_candidates)
        
        # Generate dense surface points
        dense_surface_points = []
        step_size = 1.0 / self.surface_point_density
        
        for x_float in np.arange(x_min, x_max + step_size, step_size):
            x = int(x_float)
            
            # Find the topmost point near this x-coordinate
            nearby_points = [p for p in surface_candidates if abs(p[0] - x) <= 2]
            
            if nearby_points:
                # Use the topmost point among nearby points
                topmost_y = min(p[1] for p in nearby_points)
                
                # Add sub-pixel precision if enabled
                if self.subpixel_precision and len(nearby_points) > 1:
                    # Interpolate between nearby points for sub-pixel precision
                    y_subpixel = self.interpolate_surface_y(nearby_points, x_float)
                    dense_surface_points.append((x_float, y_subpixel))
                else:
                    dense_surface_points.append((x_float, float(topmost_y)))
        
        # Step 4: Ensure we have a continuous surface line
        if len(dense_surface_points) < 5:  # Need reasonable number of points
            return []
        
        # Step 5: Remove isolated points and ensure continuity
        continuous_points = []
        for i, (x, y) in enumerate(dense_surface_points):
            # Check if this point has neighbors (continuity check)
            has_left_neighbor = i > 0 and dense_surface_points[i-1][0] >= x - 5
            has_right_neighbor = i < len(dense_surface_points)-1 and dense_surface_points[i+1][0] <= x + 5
            
            if has_left_neighbor or has_right_neighbor or len(dense_surface_points) < 10:
                continuous_points.append((x, y))
        
        # Step 6: Advanced outlier removal using statistical methods
        if len(continuous_points) > 5:
            y_coords = [p[1] for p in continuous_points]
            median_y = np.median(y_coords)
            std_y = np.std(y_coords)
            
            # Filter out points too far from median (likely container edges)
            filtered_surface = []
            for x, y in continuous_points:
                if abs(y - median_y) < 2.5 * std_y:  # Slightly more tolerance
                    filtered_surface.append((x, y))
            
            if len(filtered_surface) >= len(continuous_points) * 0.4:  # Keep if we retain at least 40%
                continuous_points = filtered_surface
        
        # Step 7: Apply advanced smoothing for better surface quality
        if len(continuous_points) > 10:
            continuous_points = self.advanced_surface_smoothing(continuous_points)
        
        # Convert back to integer coordinates for compatibility
        return [(int(x), int(y)) for x, y in continuous_points]
    
    def interpolate_surface_y(self, nearby_points, x_target):
        """Interpolate y-coordinate for a given x using nearby surface points"""
        if len(nearby_points) == 1:
            return float(nearby_points[0][1])
        
        # Sort by x-coordinate
        nearby_points.sort(key=lambda p: p[0])
        
        # Find the two closest points for interpolation
        x_coords = [p[0] for p in nearby_points]
        y_coords = [p[1] for p in nearby_points]
        
        # Linear interpolation
        if len(nearby_points) >= 2:
            return np.interp(x_target, x_coords, y_coords)
        
        return float(nearby_points[0][1])
    
    def extract_surface_line(self, contour, width):
        """Legacy method that calls the robust surface extraction"""
        return self.robust_surface_extraction(contour, width)
    
    def interpolate_surface_line(self, surface_points, width):
        """Interpolate missing points in the surface line with high density"""
        if len(surface_points) < 2:
            return surface_points
        
        # Extract x and y coordinates
        x_coords = [p[0] for p in surface_points]
        y_coords = [p[1] for p in surface_points]
        
        # Create high-density interpolation
        x_min, x_max = min(x_coords), max(x_coords)
        interpolated_points = []
        
        # Generate points with sub-pixel precision
        step_size = 1.0 / self.surface_point_density
        
        for x_float in np.arange(x_min, x_max + step_size, step_size):
            x = int(x_float)
            
            # Check if we have an exact match
            if x in x_coords:
                idx = x_coords.index(x)
                interpolated_points.append((x, y_coords[idx]))
            else:
                # Advanced interpolation based on method
                if self.interpolation_method == 'cubic' and len(surface_points) > 4:
                    # Cubic interpolation for smoother surfaces
                    y_interp = self.cubic_interpolation(x_coords, y_coords, x_float)
                elif self.interpolation_method == 'spline' and len(surface_points) > 6:
                    # Spline interpolation for very smooth surfaces
                    y_interp = self.spline_interpolation(x_coords, y_coords, x_float)
                else:
                    # Linear interpolation (default)
                    y_interp = self.linear_interpolation(x_coords, y_coords, x_float)
                
                interpolated_points.append((int(x_float), int(y_interp)))
        
        return interpolated_points
    
    def linear_interpolation(self, x_coords, y_coords, x_target):
        """Linear interpolation for surface points"""
        # Find nearest points
        left_x = max([x_c for x_c in x_coords if x_c <= x_target], default=x_coords[0])
        right_x = min([x_c for x_c in x_coords if x_c >= x_target], default=x_coords[-1])
        
        if left_x == right_x:
            return y_coords[x_coords.index(left_x)]
        
        left_y = y_coords[x_coords.index(left_x)]
        right_y = y_coords[x_coords.index(right_x)]
        
        # Linear interpolation
        return left_y + (right_y - left_y) * (x_target - left_x) / (right_x - left_x)
    
    def cubic_interpolation(self, x_coords, y_coords, x_target):
        """Cubic interpolation for smoother surface"""
        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(x_coords, y_coords)
            return cs(x_target)
        except:
            # Fall back to linear interpolation
            return self.linear_interpolation(x_coords, y_coords, x_target)
    
    def spline_interpolation(self, x_coords, y_coords, x_target):
        """Spline interpolation for very smooth surface"""
        try:
            spline = UnivariateSpline(x_coords, y_coords, s=0)
            return spline(x_target)
        except:
            # Fall back to cubic interpolation
            return self.cubic_interpolation(x_coords, y_coords, x_target)

    def calculate_surface_metrics(self, surface_points):
        """Calculate metrics for the surface line"""
        if not surface_points:
            return None
        
        y_coords = [p[1] for p in surface_points]
        
        metrics = {
            'mean_height': np.mean(y_coords),
            'std_height': np.std(y_coords),
            'min_height': np.min(y_coords),
            'max_height': np.max(y_coords),
            'height_range': np.max(y_coords) - np.min(y_coords),
            'num_points': len(surface_points)
        }
        
        return metrics

    def track_motion(self, prev_gray, curr_gray, prev_points):
        """Track motion using optical flow"""
        if len(prev_points) == 0:
            return [], []
        
        # Calculate optical flow
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None, **self.lk_params)
        
        # Select good points
        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]
        
        return good_new, good_old

    def select_roi(self, frame):
        """Allow manual ROI selection"""
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi_points = [(x, y)]
                self.selecting = True
            elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
                temp_frame = frame.copy()
                if len(self.roi_points) > 0:
                    cv2.rectangle(temp_frame, self.roi_points[0], (x, y), (0, 255, 0), 2)
                cv2.imshow('Select ROI', temp_frame)
            elif event == cv2.EVENT_LBUTTONUP:
                self.selecting = False
                if len(self.roi_points) > 0:
                    x1, y1 = self.roi_points[0]
                    self.selected_roi = (
                        min(x1, x), min(y1, y),
                        abs(x - x1), abs(y - y1)
                    )
                    self.roi_selected = True

        cv2.namedWindow('Select ROI')
        cv2.setMouseCallback('Select ROI', mouse_callback)
        
        print("ROI Selection:")
        print("1. Click and drag to select rectangular ROI")
        print("2. Press 'r' to reset selection")
        print("3. Press 'c' to confirm and continue")
        print("4. Press 'q' to quit")
        
        while True:
            temp_frame = frame.copy()
            
            # Show fluid detection preview
            fluid_mask = self.detect_colored_fluid(frame)
            fluid_preview = cv2.bitwise_and(frame, frame, mask=fluid_mask)
            
            # Overlay fluid detection
            temp_frame = cv2.addWeighted(temp_frame, 0.7, fluid_preview, 0.3, 0)
            
            # Show ROI if selected
            if self.roi_selected and self.selected_roi:
                x, y, w, h = self.selected_roi
                cv2.rectangle(temp_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(temp_frame, 'ROI Selected', (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Select ROI', temp_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):  # Reset
                self.roi_points = []
                self.selected_roi = None
                self.roi_selected = False
            elif key == ord('c'):  # Confirm
                if self.roi_selected:
                    break
                else:
                    # If no ROI selected, use automatic detection
                    print("No ROI selected, using automatic detection")
                    break
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return True

    def visualize_robust_detection(self, frame, contour, width, debug_info=None):
        """
        Visualize the robust surface detection process step by step
        
        Args:
            frame: Original frame
            contour: Fluid contour
            width: Frame width
            debug_info: Dictionary containing intermediate results
            
        Returns:
            comparison_frame: Side-by-side comparison of original vs robust detection
        """
        if contour is None:
            return frame
        
        # Create side-by-side comparison
        h, w = frame.shape[:2]
        comparison_frame = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        # Left side: Original method
        left_frame = frame.copy()
        original_points = self.extract_initial_surface_points(contour, width)
        
        if original_points:
            # Draw original surface points in red
            for point in original_points:
                cv2.circle(left_frame, point, 2, (0, 0, 255), -1)
            
            # Draw original surface line
            if len(original_points) > 1:
                points = np.array(original_points, dtype=np.int32)
                cv2.polylines(left_frame, [points], False, (0, 0, 255), 2)
        
        cv2.putText(left_frame, 'Original Detection', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(left_frame, f'Points: {len(original_points)}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Right side: Robust method
        right_frame = frame.copy()
        robust_points = self.robust_surface_extraction(contour, width)
        
        if robust_points:
            # Draw robust surface points in green
            for point in robust_points:
                cv2.circle(right_frame, point, 2, (0, 255, 0), -1)
            
            # Draw robust surface line
            if len(robust_points) > 1:
                points = np.array(robust_points, dtype=np.int32)
                cv2.polylines(right_frame, [points], False, (0, 255, 0), 2)
        
        cv2.putText(right_frame, 'Robust Detection', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(right_frame, f'Points: {len(robust_points)}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add processing info
        if debug_info:
            y_offset = 90
            for key, value in debug_info.items():
                cv2.putText(right_frame, f'{key}: {value}', (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20
        
        # Combine frames
        comparison_frame[:, :w] = left_frame
        comparison_frame[:, w:] = right_frame
        
        # Draw separator line
        cv2.line(comparison_frame, (w, 0), (w, h), (255, 255, 255), 2)
        
        return comparison_frame
    
    def debug_surface_detection(self, frame, fluid_mask):
        """Debug surface detection step by step"""
        h, w = frame.shape[:2]
        
        # Create debug visualization
        debug_frame = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        # Panel 1: Original frame with fluid mask
        panel1 = frame.copy()
        fluid_overlay = cv2.bitwise_and(frame, frame, mask=fluid_mask)
        panel1 = cv2.addWeighted(panel1, 0.7, fluid_overlay, 0.3, 0)
        
        # Panel 2: Contour-based detection
        panel2 = frame.copy()
        contour_points = self.extract_surface_from_contours(fluid_mask, w)
        if contour_points:
            for point in contour_points:
                cv2.circle(panel2, point, 3, (0, 255, 0), -1)
            if len(contour_points) > 1:
                points = np.array(contour_points, dtype=np.int32)
                cv2.polylines(panel2, [points], False, (0, 255, 0), 2)
        
        # Panel 3: Mask-based detection
        panel3 = frame.copy()
        mask_points = self.extract_surface_from_mask(fluid_mask, w)
        if mask_points:
            for point in mask_points:
                cv2.circle(panel3, point, 3, (0, 0, 255), -1)
            if len(mask_points) > 1:
                points = np.array(mask_points, dtype=np.int32)
                cv2.polylines(panel3, [points], False, (0, 0, 255), 2)
        
        # Add labels
        cv2.putText(panel1, 'Fluid Mask', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel2, f'Contour Method ({len(contour_points)} pts)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(panel3, f'Mask Method ({len(mask_points)} pts)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Combine panels
        debug_frame[:, :w] = panel1
        debug_frame[:, w:2*w] = panel2
        debug_frame[:, 2*w:] = panel3
        
        # Draw separator lines
        cv2.line(debug_frame, (w, 0), (w, h), (255, 255, 255), 2)
        cv2.line(debug_frame, (2*w, 0), (2*w, h), (255, 255, 255), 2)
        
        return debug_frame
    
    def visualize_analysis(self, frame, surface_points, motion_vectors, frame_num, fluid_mask):
        """Visualize the analysis results with robust detection information"""
        vis_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Show fluid detection overlay
        fluid_colored = cv2.bitwise_and(frame, frame, mask=fluid_mask)
        vis_frame = cv2.addWeighted(vis_frame, 0.8, fluid_colored, 0.2, 0)
        
        # Draw surface line with different colors for different methods
        if surface_points and len(surface_points) > 1:
            # Convert to numpy array for polylines
            points = np.array(surface_points, dtype=np.int32)
            
            # Use different colors based on detection method
            line_color = (0, 255, 0) if self.use_robust_detection else (0, 0, 255)
            cv2.polylines(vis_frame, [points], False, line_color, 3)
            
            # Draw surface points
            for point in surface_points[::5]:  # Show every 5th point to avoid clutter
                cv2.circle(vis_frame, point, 3, (0, 255, 255), -1)
        
        # Draw motion vectors
        for i, (new, old) in enumerate(motion_vectors):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            
            # Draw motion line
            cv2.line(vis_frame, (a, b), (c, d), (255, 0, 0), 2)
            cv2.circle(vis_frame, (a, b), 3, (0, 0, 255), -1)
        
        # Add information text
        cv2.putText(vis_frame, f'Frame: {frame_num}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f'Surface Points: {len(surface_points)}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, f'Motion Vectors: {len(motion_vectors)}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show analysis mode
        mode_text = "Full Frame" if (self.roi_selected and self.selected_roi == (0, 0, w, h)) else "ROI Mode"
        cv2.putText(vis_frame, f'Mode: {mode_text}', (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show detection method
        detection_method = "Robust (RANSAC+Kalman)" if self.use_robust_detection else "Standard"
        cv2.putText(vis_frame, f'Detection: {detection_method}', (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show surface point density and interpolation method
        cv2.putText(vis_frame, f'Density: {self.surface_point_density}x | {self.interpolation_method}', (10, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add surface metrics if available
        if surface_points:
            metrics = self.calculate_surface_metrics(surface_points)
            if metrics:
                cv2.putText(vis_frame, f'Mean Height: {metrics["mean_height"]:.1f}', (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_frame, f'Height Range: {metrics["height_range"]:.1f}', (10, 210), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add motion speed information
                cv2.putText(vis_frame, f'Motion Speed: {self.motion_speed:.1f}', (10, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Indicate fast motion mode
                if self.motion_speed > self.fast_motion_threshold:
                    cv2.putText(vis_frame, 'FAST MOTION', (10, 270), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(vis_frame, 'Normal Motion', (10, 270), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show detection quality and confidence
                if hasattr(self, 'detection_confidence'):
                    confidence_pct = self.detection_confidence * 100
                    confidence_color = (0, 255, 0) if confidence_pct > 80 else (0, 255, 255) if confidence_pct > 50 else (0, 0, 255)
                    cv2.putText(vis_frame, f'Confidence: {confidence_pct:.1f}%', (10, 330), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, confidence_color, 2)
                
                # Show current detection quality
                if surface_points:
                    quality = self.calculate_detection_quality(surface_points)
                    quality_pct = quality * 100
                    quality_color = (0, 255, 0) if quality_pct > 80 else (0, 255, 255) if quality_pct > 50 else (0, 0, 255)
                    cv2.putText(vis_frame, f'Quality: {quality_pct:.1f}%', (10, 360), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
        
        # Draw ROI if selected
        if self.roi_selected and self.selected_roi:
            x, y, w_roi, h_roi = self.selected_roi
            cv2.rectangle(vis_frame, (x, y), (x + w_roi, y + h_roi), (255, 0, 255), 2)
        
        return vis_frame

    def adapt_to_lighting(self, frame):
        """Automatically adapt color parameters based on lighting conditions and detection quality"""
        if not self.color_selected or self.target_color_hsv is None:
            return
        
        # Analyze overall lighting conditions
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get average brightness and saturation
        avg_brightness = np.mean(hsv[:, :, 2])  # V channel
        avg_saturation = np.mean(hsv[:, :, 1])  # S channel
        
        # Adjust tolerance based on lighting conditions
        base_adjustment = 1.0
        
        # Consider detection confidence for adjustment strength
        if hasattr(self, 'detection_confidence'):
            if self.detection_confidence < 0.5:
                base_adjustment = 1.5  # More aggressive adjustment for poor detection
            elif self.detection_confidence < 0.8:
                base_adjustment = 1.2  # Moderate adjustment
        
        if avg_brightness < 100:  # Dark conditions
            self.value_tolerance = min(140, self.value_tolerance * (1.2 * base_adjustment))
            self.min_value = max(15, self.min_value * (0.8 / base_adjustment))
        elif avg_brightness > 200:  # Bright conditions
            self.value_tolerance = max(50, self.value_tolerance * (0.8 / base_adjustment))
            self.min_value = min(60, self.min_value * (1.2 * base_adjustment))
        
        if avg_saturation < 80:  # Low saturation conditions
            self.saturation_tolerance = min(120, self.saturation_tolerance * (1.3 * base_adjustment))
            self.min_saturation = max(15, self.min_saturation * (0.7 / base_adjustment))
        
        # Adaptive hue tolerance based on detection quality
        if hasattr(self, 'detection_confidence') and self.detection_confidence < 0.6:
            self.hue_tolerance = min(50, self.hue_tolerance * 1.3)
        
        # Recalculate adaptive ranges if we have samples
        if self.color_samples:
            self.calculate_adaptive_color_ranges()
    
    def debug_color_detection(self, frame):
        """Debug method to help tune color detection parameters"""
        # Get the fluid mask
        fluid_mask = self.detect_colored_fluid(frame)
        
        # Show different stages of detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if self.color_selected and self.adaptive_hsv_ranges:
            # Show adaptive color detection
            mask1 = cv2.inRange(hsv, self.adaptive_hsv_ranges['lower'], self.adaptive_hsv_ranges['upper'])
            combined_mask = mask1.copy()
            
            if 'lower2' in self.adaptive_hsv_ranges:
                mask2 = cv2.inRange(hsv, self.adaptive_hsv_ranges['lower2'], self.adaptive_hsv_ranges['upper2'])
                combined_mask = cv2.bitwise_or(mask1, mask2)
        else:
            # Show default color detection
            mask_pink1 = cv2.inRange(hsv, self.lower_pink1, self.upper_pink1)
            mask_pink2 = cv2.inRange(hsv, self.lower_pink2, self.upper_pink2)
            combined_mask = cv2.bitwise_or(mask_pink1, mask_pink2)
        
        # Create visualization
        debug_frame = frame.copy()
        
        # Highlight detected fluid areas
        fluid_colored = cv2.bitwise_and(frame, frame, mask=fluid_mask)
        debug_frame = cv2.addWeighted(debug_frame, 0.6, fluid_colored, 0.4, 0)
        
        # Draw border exclusion zone
        cv2.rectangle(debug_frame, 
                     (self.edge_margin, self.edge_margin), 
                     (frame.shape[1] - self.edge_margin, frame.shape[0] - self.edge_margin), 
                     (255, 0, 0), 2)
        
        # Add text information
        detection_mode = "Adaptive" if self.color_selected else "Default"
        cv2.putText(debug_frame, f'Detection Mode: {detection_mode}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_frame, 'Blue: Edge exclusion zone', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(debug_frame, 'Colored areas: Detected fluid', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.color_selected:
            cv2.putText(debug_frame, f'Target HSV: {self.target_color_hsv}', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return debug_frame, fluid_mask, combined_mask

    def analyze_wave_motion(self, debug_mode=False, use_full_frame=False):
        """Analyze fluid surface motion in the video"""
        if not self.cap.isOpened():
            print("Error: Cannot open video file")
            return
        
        # Read first frame
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Cannot read first frame")
            return
        
        # Handle ROI selection based on mode
        if use_full_frame:
            # Use entire frame as ROI
            h, w = frame.shape[:2]
            self.selected_roi = (0, 0, w, h)
            self.roi_selected = True
            print(f"Using full frame as ROI: {w}x{h}")
        else:
            # Allow manual ROI selection
            if not self.select_roi(frame):
                print("ROI selection cancelled")
                return
        
        # Try to load existing color profile
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        
        if self.load_color_profile(video_name):
            print("Existing color profile loaded. Press 'r' to reselect colors if needed.")
        else:
            # Allow fluid color selection
            if not self.select_fluid_color(frame):
                print("Color selection cancelled")
                return
            
            # Save the color profile for future use
            self.save_color_profile(video_name)
        
        # Initialize variables
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_num = 0
        
        # Setup video writer if output is requested
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
        
        # Initial surface detection
        surface_points, fluid_mask = self.find_fluid_surface_line(frame)
        prev_points = None
        
        if surface_points:
            # Convert surface points to optical flow format
            prev_points = np.array(surface_points, dtype=np.float32).reshape(-1, 1, 2)
            
            # Store initial surface metrics
            initial_metrics = self.calculate_surface_metrics(surface_points)
            self.surface_positions.append(initial_metrics)
        
        print("Starting fluid motion analysis...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'p' - Pause/Resume")
        print("  's' - Save current frame")
        print("  'r' - Reset tracking points")
        print("  'd' - Toggle debug mode")
        print("  'c' - Reselect fluid color")
        print("  'b' - Toggle robust detection (RANSAC+Kalman)")
        print("  'v' - Toggle comparison view (original vs robust)")
        print("  'x' - Debug surface detection step by step")
        print("  '+' - Increase surface point density")
        print("  '-' - Decrease surface point density")
        print("  'i' - Toggle interpolation method (linear/cubic/spline)")
        print("  'f' - Force detection recovery (reset thresholds and cache)")
        print("  'a' - Toggle adaptive threshold adjustment")
        
        paused = False
        comparison_mode = False
        surface_debug_mode = False
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_num += 1
                
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Adapt to lighting conditions every 30 frames
                if frame_num % 30 == 0:
                    self.adapt_to_lighting(frame)
                
                # Check for automatic recovery every 60 frames
                if frame_num % 60 == 0:
                    self.check_automatic_recovery()
                
                # Detect surface line
                surface_points, fluid_mask = self.find_fluid_surface_line(frame)
                
                # Calculate and store surface metrics
                if surface_points:
                    metrics = self.calculate_surface_metrics(surface_points)
                    self.surface_positions.append(metrics)
                else:
                    self.surface_positions.append(None)
                
                # Track motion
                motion_vectors = []
                if prev_points is not None and len(prev_points) > 0:
                    good_new, good_old = self.track_motion(prev_gray, curr_gray, prev_points)
                    motion_vectors = list(zip(good_new, good_old))
                    
                    # Update points for next frame
                    if len(good_new) > 0:
                        prev_points = good_new.reshape(-1, 1, 2)
                    else:
                        # Reset tracking points from surface
                        if surface_points:
                            prev_points = np.array(surface_points, dtype=np.float32).reshape(-1, 1, 2)
                        else:
                            prev_points = None
                
                # Store surface history
                self.surface_history.append(surface_points)
                
                prev_gray = curr_gray.copy()
            
            # Visualize results
            if surface_debug_mode:
                # Show surface detection debug
                debug_frame = self.debug_surface_detection(frame, fluid_mask)
                cv2.imshow('Surface Detection Debug', debug_frame)
            elif comparison_mode:
                # Show side-by-side comparison of original vs robust detection
                contours, _ = cv2.findContours(fluid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                main_contour = None
                if contours:
                    main_contour = max(contours, key=cv2.contourArea)
                
                vis_frame = self.visualize_robust_detection(frame, main_contour, w)
                cv2.imshow('Robust Detection Comparison', vis_frame)
            else:
                if debug_mode:
                    debug_frame, fluid_mask_debug, combined_mask = self.debug_color_detection(frame)
                    vis_frame = self.visualize_analysis(debug_frame, surface_points, motion_vectors, frame_num, fluid_mask)
                    
                    # Show debug information
                    cv2.imshow('Color Detection Debug', combined_mask)
                    cv2.imshow('Refined Fluid Mask', fluid_mask)
                else:
                    vis_frame = self.visualize_analysis(frame, surface_points, motion_vectors, frame_num, fluid_mask)
                
                # Show analysis
                cv2.imshow('Fluid Motion Analysis', vis_frame)
            
            # Save video if requested
            if self.output_path and not paused:
                out.write(vis_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('s'):
                cv2.imwrite(f'frame_{frame_num}.jpg', vis_frame)
                print(f"Frame {frame_num} saved")
            elif key == ord('r'):
                # Reset tracking points
                if surface_points:
                    prev_points = np.array(surface_points, dtype=np.float32).reshape(-1, 1, 2)
                print("Tracking points reset")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
                if not debug_mode:
                    cv2.destroyWindow('Color Detection Debug')
                    cv2.destroyWindow('Refined Fluid Mask')
            elif key == ord('c'):
                # Reselect fluid color
                cv2.destroyWindow('Fluid Motion Analysis')
                if self.select_fluid_color(frame):
                    self.save_color_profile(video_name)
                    print("Color profile updated and saved")
                else:
                    print("Color reselection cancelled")
                cv2.namedWindow('Fluid Motion Analysis')
            elif key == ord('b'):
                # Toggle robust detection
                self.use_robust_detection = not self.use_robust_detection
                print(f"Robust detection: {'ON' if self.use_robust_detection else 'OFF'}")
                if self.use_robust_detection:
                    print("  - RANSAC outlier removal: ENABLED")
                    print("  - Kalman temporal smoothing: ENABLED")
                    print("  - Physics constraints: ENABLED")
                else:
                    print("  - Using standard detection method")
            elif key == ord('v'):
                # Toggle comparison view
                comparison_mode = not comparison_mode
                print(f"Comparison view: {'ON' if comparison_mode else 'OFF'}")
                if comparison_mode:
                    cv2.destroyWindow('Fluid Motion Analysis')
                    if debug_mode:
                        cv2.destroyWindow('Color Detection Debug')
                        cv2.destroyWindow('Refined Fluid Mask')
                    if surface_debug_mode:
                        cv2.destroyWindow('Surface Detection Debug')
                        surface_debug_mode = False
                else:
                    cv2.destroyWindow('Robust Detection Comparison')
            elif key == ord('x'):
                # Toggle surface detection debug
                surface_debug_mode = not surface_debug_mode
                print(f"Surface detection debug: {'ON' if surface_debug_mode else 'OFF'}")
                if surface_debug_mode:
                    cv2.destroyWindow('Fluid Motion Analysis')
                    if debug_mode:
                        cv2.destroyWindow('Color Detection Debug')
                        cv2.destroyWindow('Refined Fluid Mask')
                    if comparison_mode:
                        cv2.destroyWindow('Robust Detection Comparison')
                        comparison_mode = False
                else:
                    cv2.destroyWindow('Surface Detection Debug')
            elif key == ord('+'):
                # Increase surface point density
                self.surface_point_density = min(5, self.surface_point_density + 0.5)
                print(f"Surface point density increased to: {self.surface_point_density}")
            elif key == ord('-'):
                # Decrease surface point density
                self.surface_point_density = max(0.5, self.surface_point_density - 0.5)
                print(f"Surface point density decreased to: {self.surface_point_density}")
            elif key == ord('i'):
                # Toggle interpolation method
                methods = ['linear', 'cubic', 'spline']
                current_idx = methods.index(self.interpolation_method)
                self.interpolation_method = methods[(current_idx + 1) % len(methods)]
                print(f"Interpolation method changed to: {self.interpolation_method}")
            elif key == ord('f'):
                # Force detection recovery
                self.force_detection_recovery()
                print("Detection recovery forced - thresholds reset and cache cleared")
            elif key == ord('a'):
                # Toggle adaptive threshold adjustment
                self.adaptive_threshold_adjustment = not self.adaptive_threshold_adjustment
                print(f"Adaptive threshold adjustment: {'ON' if self.adaptive_threshold_adjustment else 'OFF'}")
            
            # Progress indicator
            if frame_num % 30 == 0 and not paused:
                print(f"Processed frame {frame_num}")
        
        # Cleanup
        self.cap.release()
        if self.output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Analysis completed. Processed {frame_num} frames.")
        print(f"Surface data points collected: {len(self.surface_positions)}")
        
        # Print detection statistics
        valid_detections = sum(1 for pos in self.surface_positions if pos is not None)
        if len(self.surface_positions) > 0:
            success_rate = valid_detections / len(self.surface_positions) * 100
            print(f"Surface detection success rate: {success_rate:.1f}%")
            
            if success_rate < 80:
                print("TIP: Low detection rate. Consider:")
                print("  - Adjusting color tolerance in the code")
                print("  - Selecting a more specific ROI")
                print("  - Checking if the fluid color matches RGB(200, 79, 114)")

    def plot_surface_analysis(self):
        """Create comprehensive surface analysis plots"""
        if not self.surface_positions:
            print("No surface data available for analysis")
            return
        
        # Extract metrics over time
        frames = range(len(self.surface_positions))
        mean_heights = []
        height_ranges = []
        std_heights = []
        
        for metrics in self.surface_positions:
            if metrics:
                mean_heights.append(metrics['mean_height'])
                height_ranges.append(metrics['height_range'])
                std_heights.append(metrics['std_height'])
            else:
                mean_heights.append(np.nan)
                height_ranges.append(np.nan)
                std_heights.append(np.nan)
        
        # Create comprehensive plots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Mean surface height over time
        plt.subplot(2, 2, 1)
        plt.plot(frames, mean_heights, 'b-', linewidth=2)
        plt.title('Mean Surface Height Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Height (pixels)')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Surface height variation (range)
        plt.subplot(2, 2, 2)
        plt.plot(frames, height_ranges, 'r-', linewidth=2)
        plt.title('Surface Height Range Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Height Range (pixels)')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Surface stability (standard deviation)
        plt.subplot(2, 2, 3)
        plt.plot(frames, std_heights, 'g-', linewidth=2)
        plt.title('Surface Stability (Standard Deviation)')
        plt.xlabel('Frame')
        plt.ylabel('Standard Deviation (pixels)')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Movement velocity (change in mean height)
        plt.subplot(2, 2, 4)
        if len(mean_heights) > 1:
            # Calculate velocity as difference between consecutive frames
            velocities = np.diff(mean_heights)
            plt.plot(frames[1:], velocities, 'm-', linewidth=2)
            plt.title('Surface Movement Velocity')
            plt.xlabel('Frame')
            plt.ylabel('Velocity (pixels/frame)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        valid_heights = [h for h in mean_heights if not np.isnan(h)]
        if valid_heights:
            print("\n=== SURFACE ANALYSIS SUMMARY ===")
            print(f"Total frames analyzed: {len(self.surface_positions)}")
            print(f"Frames with valid surface detection: {len(valid_heights)}")
            print(f"Detection success rate: {len(valid_heights)/len(self.surface_positions)*100:.1f}%")
            print(f"Mean surface height: {np.mean(valid_heights):.2f} pixels")
            print(f"Surface height standard deviation: {np.std(valid_heights):.2f} pixels")
            print(f"Maximum surface movement: {np.max(valid_heights) - np.min(valid_heights):.2f} pixels")
            
            if len(valid_heights) > 1:
                velocities = np.diff(valid_heights)
                print(f"Mean movement velocity: {np.mean(np.abs(velocities)):.2f} pixels/frame")
                print(f"Maximum movement velocity: {np.max(np.abs(velocities)):.2f} pixels/frame")

# Example usage
def main():
    # Available video files in the directory
    video_files = [
        "imapttiG_1_prima parte.mp4",
        "imapttiG_1_seconda_parte.mp4", 
        "MicroG_Impatto_4.mp4",
        "colorante.mp4",
        "exported.mp4",
        "WhatsApp Video 2025-06-02 at 11.32.47.mp4"
    ]
    
    # Let user choose video or use default
    print("=== FLUID SURFACE TRACKER ===")
    print("Available videos:")
    for i, video in enumerate(video_files):
        if os.path.exists(video):
            print(f"{i+1}. {video}")
        else:
            print(f"{i+1}. {video} (NOT FOUND)")
    
    print("\nChoose a video number (or press Enter for default):")
    choice = input().strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(video_files):
        video_path = video_files[int(choice) - 1]
    else:
        video_path = "imapttiG_1_prima parte.mp4"  # Default
    
    if not os.path.exists(video_path):
        print(f"ERROR: File {video_path} does not exist.")
        print("Please make sure the video file is in the current directory.")
        return
    
    # Create output path
    base_name = os.path.splitext(video_path)[0]
    output_path = f"analisi_fluido_output_{base_name}.mp4"
    
    print(f"\nSelected video: {video_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Create tracker
    tracker = FluidTracker(video_path, output_path)
    
    # Mode selection
    print("\n=== ANALYSIS MODE SELECTION ===")
    print("Choose analysis mode:")
    print("1. Full Frame Mode - Use entire video frame (no ROI selection)")
    print("2. ROI Mode - Manually select Region of Interest")
    
    while True:
        mode_choice = input("\nEnter mode (1 or 2): ").strip()
        if mode_choice == "1":
            use_full_frame = True
            print("Selected: Full Frame Mode")
            break
        elif mode_choice == "2":
            use_full_frame = False
            print("Selected: ROI Mode")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    # Run analysis with debug mode enabled for better tuning
    print("\n=== STARTING ANALYSIS ===")
    if use_full_frame:
        print("1. Using full frame as analysis area")
        print("2. You'll select fluid color by clicking on it")
        print("3. Finally the analysis will start")
    else:
        print("1. First you'll select ROI (Region of Interest)")
        print("2. Then you'll select fluid color by clicking on it")
        print("3. Finally the analysis will start")
    
    tracker.analyze_wave_motion(debug_mode=True, use_full_frame=use_full_frame)
    
    # Show analysis plots
    tracker.plot_surface_analysis()

def quick_test(video_path, use_full_frame=False):
    """Quick test function for a specific video"""
    if not os.path.exists(video_path):
        print(f"ERROR: File {video_path} does not exist.")
        return
    
    base_name = os.path.splitext(video_path)[0]
    output_path = f"analisi_fluido_output_{base_name}.mp4"
    
    tracker = FluidTracker(video_path, output_path)
    tracker.analyze_wave_motion(debug_mode=True, use_full_frame=use_full_frame)
    tracker.plot_surface_analysis()

if __name__ == "__main__":
    main()