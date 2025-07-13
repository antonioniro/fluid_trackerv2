import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

class FluidTracker:
    def __init__(self, video_path, output_path=None):
        self.cap = cv2.VideoCapture(video_path)
        self.output_path = output_path
        self.surface_history = deque(maxlen=100)  # Increased history for better analysis
        self.motion_vectors = []
        self.surface_positions = []  # Track surface position over time
        
        # Color detection parameters for specific pink fluid (RGB: 200, 79, 114)
        # Convert RGB to HSV: R=200, G=79, B=114 -> H≈338, S≈60, V≈78
        self.lower_pink1 = np.array([160, 40, 50])   # Lower HSV range for pink (330-360° in H)
        self.upper_pink1 = np.array([180, 255, 255]) # Upper HSV range for pink
        self.lower_pink2 = np.array([0, 40, 50])     # Lower HSV range for pink (0-20° in H)
        self.upper_pink2 = np.array([20, 255, 255])  # Upper HSV range for pink
        
        # More specific color detection for better precision
        self.color_tolerance = 30  # Tolerance for color matching
        self.min_saturation = 40   # Minimum saturation to avoid detecting container
        self.min_value = 50        # Minimum value to avoid shadows
        
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
        self.min_contour_area = 1000  # Increased minimum area for fluid contour
        self.smoothing_window = 7     # Increased window for better smoothing
        self.max_surface_jump = 50    # Maximum allowed jump in surface position
        
        # Container edge filtering
        self.edge_margin = 20         # Margin from image edges to ignore
        self.min_fluid_width = 100    # Minimum width of fluid surface to consider valid
        
    def detect_colored_fluid(self, frame):
        """Detect pink colored fluid using precise color thresholding"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        
        # Create masks for pink color ranges (handles hue wrap-around)
        mask_pink1 = cv2.inRange(hsv, self.lower_pink1, self.upper_pink1)
        mask_pink2 = cv2.inRange(hsv, self.lower_pink2, self.upper_pink2)
        
        # Combine masks
        mask = cv2.bitwise_or(mask_pink1, mask_pink2)
        
        # Additional filtering to exclude container edges
        # Create a mask to exclude image borders (likely container edges)
        border_mask = np.ones_like(mask)
        border_mask[:self.edge_margin, :] = 0  # Top edge
        border_mask[-self.edge_margin:, :] = 0  # Bottom edge
        border_mask[:, :self.edge_margin] = 0   # Left edge
        border_mask[:, -self.edge_margin:] = 0  # Right edge
        
        mask = cv2.bitwise_and(mask, border_mask)
        
        # Apply morphological operations to clean up the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)  # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)  # Fill gaps
        
        # Apply additional filtering based on color similarity to target RGB
        target_rgb = np.array([200, 79, 114])  # Target fluid color
        
        # Create a more refined mask based on color distance
        refined_mask = np.zeros_like(mask)
        
        # Convert frame to RGB for color distance calculation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Only process areas where the initial mask is positive
        mask_indices = np.where(mask > 0)
        
        if len(mask_indices[0]) > 0:
            # Calculate color distances for pixels in the mask
            pixels = frame_rgb[mask_indices]
            distances = np.sqrt(np.sum((pixels - target_rgb)**2, axis=1))
            
            # Keep only pixels within color tolerance
            valid_pixels = distances < self.color_tolerance
            refined_mask[mask_indices[0][valid_pixels], mask_indices[1][valid_pixels]] = 255
        
        # Apply final smoothing
        refined_mask = cv2.GaussianBlur(refined_mask, (3, 3), 0)
        
        return refined_mask
    
    def find_fluid_surface_line(self, frame):
        """Find the surface line of the colored fluid with improved filtering"""
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
                        x + w_cont < w - self.edge_margin and
                        y > self.edge_margin and 
                        w_cont > self.min_fluid_width):
                        valid_contours.append(contour)
            
            if valid_contours:
                # Use the largest valid contour (main fluid body)
                largest_contour = max(valid_contours, key=cv2.contourArea)
                
                # Extract surface line from the main fluid contour
                surface_points = self.extract_surface_line(largest_contour, w)
                
                # Additional validation: check if surface line looks reasonable
                if surface_points:
                    # Ensure surface line is not too fragmented
                    x_coords = [p[0] for p in surface_points]
                    if len(x_coords) > 1:
                        x_range = max(x_coords) - min(x_coords)
                        # If the surface line is too fragmented, reject it
                        if x_range < self.min_fluid_width:
                            surface_points = []
        
        return surface_points, fluid_mask
    
    def extract_surface_line(self, contour, width):
        """Extract the surface line from the fluid contour with improved filtering"""
        # Get all points from the contour
        contour_points = contour.reshape(-1, 2)
        
        # Filter out points too close to image edges (likely container edges)
        filtered_points = []
        for point in contour_points:
            x, y = point
            if (self.edge_margin < x < width - self.edge_margin and 
                y > self.edge_margin):  # Allow points near top but not other edges
                filtered_points.append(point)
        
        if len(filtered_points) < 3:  # Need at least 3 points for a line
            return []
        
        # Group points by x-coordinate and find the minimum y for each x (topmost points)
        surface_dict = {}
        for point in filtered_points:
            x, y = point
            if x not in surface_dict or y < surface_dict[x]:
                surface_dict[x] = y
        
        # Convert to sorted list of surface points
        surface_points = [(x, y) for x, y in sorted(surface_dict.items())]
        
        # Filter out isolated points (likely noise or container edges)
        if len(surface_points) < self.min_fluid_width // 10:  # Need minimum density
            return []
        
        # Remove outliers that are too far from the median height
        if len(surface_points) > 5:
            y_coords = [p[1] for p in surface_points]
            median_y = np.median(y_coords)
            std_y = np.std(y_coords)
            
            # Filter out points too far from median (likely container edges)
            filtered_surface = []
            for x, y in surface_points:
                if abs(y - median_y) < 2 * std_y:  # Within 2 standard deviations
                    filtered_surface.append((x, y))
            
            if len(filtered_surface) >= len(surface_points) * 0.5:  # Keep if we retain at least 50%
                surface_points = filtered_surface
        
        # Apply smoothing to reduce noise
        if len(surface_points) > self.smoothing_window:
            smoothed_points = []
            for i in range(len(surface_points)):
                start_idx = max(0, i - self.smoothing_window // 2)
                end_idx = min(len(surface_points), i + self.smoothing_window // 2 + 1)
                
                window_points = surface_points[start_idx:end_idx]
                avg_y = sum(p[1] for p in window_points) / len(window_points)
                smoothed_points.append((surface_points[i][0], int(avg_y)))
            
            surface_points = smoothed_points
        
        # Check if the surface line is continuous enough
        if len(surface_points) > 1:
            x_coords = [p[0] for p in surface_points]
            x_span = max(x_coords) - min(x_coords)
            
            # Only keep surface if it spans a reasonable width
            if x_span >= self.min_fluid_width:
                # Interpolate missing points to create a continuous line
                surface_points = self.interpolate_surface_line(surface_points, width)
            else:
                return []  # Surface too narrow, likely container edge
        
        return surface_points
    
    def interpolate_surface_line(self, surface_points, width):
        """Interpolate missing points in the surface line"""
        if len(surface_points) < 2:
            return surface_points
        
        # Extract x and y coordinates
        x_coords = [p[0] for p in surface_points]
        y_coords = [p[1] for p in surface_points]
        
        # Create interpolation for missing x values
        x_min, x_max = min(x_coords), max(x_coords)
        interpolated_points = []
        
        for x in range(x_min, x_max + 1):
            if x in x_coords:
                # Use existing point
                idx = x_coords.index(x)
                interpolated_points.append((x, y_coords[idx]))
            else:
                # Interpolate y value
                # Find nearest points
                left_x = max([x_c for x_c in x_coords if x_c < x], default=x_coords[0])
                right_x = min([x_c for x_c in x_coords if x_c > x], default=x_coords[-1])
                
                if left_x == right_x:
                    y = y_coords[x_coords.index(left_x)]
                else:
                    left_y = y_coords[x_coords.index(left_x)]
                    right_y = y_coords[x_coords.index(right_x)]
                    # Linear interpolation
                    y = int(left_y + (right_y - left_y) * (x - left_x) / (right_x - left_x))
                
                interpolated_points.append((x, y))
        
        return interpolated_points

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

    def visualize_analysis(self, frame, surface_points, motion_vectors, frame_num, fluid_mask):
        """Visualize the analysis results"""
        vis_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Show fluid detection overlay
        fluid_colored = cv2.bitwise_and(frame, frame, mask=fluid_mask)
        vis_frame = cv2.addWeighted(vis_frame, 0.8, fluid_colored, 0.2, 0)
        
        # Draw surface line
        if surface_points and len(surface_points) > 1:
            # Convert to numpy array for polylines
            points = np.array(surface_points, dtype=np.int32)
            cv2.polylines(vis_frame, [points], False, (0, 255, 0), 3)
            
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
        
        # Add surface metrics if available
        if surface_points:
            metrics = self.calculate_surface_metrics(surface_points)
            if metrics:
                cv2.putText(vis_frame, f'Mean Height: {metrics["mean_height"]:.1f}', (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_frame, f'Height Range: {metrics["height_range"]:.1f}', (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw ROI if selected
        if self.roi_selected and self.selected_roi:
            x, y, w_roi, h_roi = self.selected_roi
            cv2.rectangle(vis_frame, (x, y), (x + w_roi, y + h_roi), (255, 0, 255), 2)
        
        return vis_frame

    def debug_color_detection(self, frame):
        """Debug method to help tune color detection parameters"""
        # Get the fluid mask
        fluid_mask = self.detect_colored_fluid(frame)
        
        # Show different stages of detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Individual color masks
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
        cv2.putText(debug_frame, 'Blue: Edge exclusion zone', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(debug_frame, 'Colored areas: Detected fluid', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return debug_frame, fluid_mask, combined_mask

    def analyze_wave_motion(self, debug_mode=False):
        """Analyze fluid surface motion in the video"""
        if not self.cap.isOpened():
            print("Error: Cannot open video file")
            return
        
        # Read first frame
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Cannot read first frame")
            return
        
        # Allow manual ROI selection
        if not self.select_roi(frame):
            print("ROI selection cancelled")
            return
        
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
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_num += 1
                
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
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
    # Replace with your video path
    video_path = "MicroG_Impatto_4.mp4"  # Change this path
    output_path = "analisi_fluido_output_microG_4.mp4"  # Optional: to save analyzed video
    
    if not os.path.exists(video_path):
        print(f"WARNING: File {video_path} does not exist.")
        print("Please modify the video path in the main() function")
        return
    
    # Create tracker
    tracker = FluidTracker(video_path, output_path)
    
    # Run analysis with debug mode enabled for better tuning
    tracker.analyze_wave_motion(debug_mode=True)
    
    # Show analysis plots
    tracker.plot_surface_analysis()

if __name__ == "__main__":
    main()