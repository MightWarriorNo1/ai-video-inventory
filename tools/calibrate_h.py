"""
Homography Calibration Tool

Interactive tool for calibrating camera homography (image -> world coordinates).
Click 4+ landmark points on an image and enter corresponding world coordinates.
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path


class HomographyCalibrator:
    """
    Interactive homography calibration tool.
    """
    
    def __init__(self, image_path: str):
        """
        Initialize calibrator.
        
        Args:
            image_path: Path to calibration image
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        self.image_points = []
        self.world_points = []
        self.window_name = "Homography Calibration - Click 4+ points, press 'q' when done"
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for clicking points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_points.append([x, y])
            print(f"Point {len(self.image_points)}: Image ({x}, {y})")
            
            # Prompt for world coordinates
            try:
                world_x = float(input(f"  Enter world X (meters): "))
                world_y = float(input(f"  Enter world Y (meters): "))
                self.world_points.append([world_x, world_y])
                print(f"  World ({world_x}, {world_y})")
            except ValueError:
                print("  Invalid input, skipping point")
                self.image_points.pop()
                return
            
            # Draw point on image
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(
                self.image,
                f"{len(self.image_points)}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.imshow(self.window_name, self.image)
    
    def calibrate(self) -> tuple:
        """
        Run interactive calibration.
        
        Returns:
            Tuple of (homography_matrix, rmse)
        """
        print("\n=== Homography Calibration ===")
        print("Instructions:")
        print("1. Click 4+ landmark points on the image")
        print("2. Enter corresponding world coordinates (in meters)")
        print("3. Press 'q' when done (minimum 4 points required)")
        print()
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Display image
        cv2.imshow(self.window_name, self.image)
        
        # Wait for 'q' key
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        # Check minimum points
        if len(self.image_points) < 4:
            raise ValueError("Need at least 4 points for homography calibration")
        
        # Convert to numpy arrays
        img_pts = np.array(self.image_points, dtype=np.float32)
        world_pts = np.array(self.world_points, dtype=np.float32)
        
        # Compute homography
        H, mask = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            raise ValueError("Failed to compute homography")
        
        # Compute RMSE
        projected = cv2.perspectiveTransform(
            img_pts.reshape(-1, 1, 2),
            H
        ).reshape(-1, 2)
        
        errors = np.sqrt(np.sum((projected - world_pts) ** 2, axis=1))
        rmse = np.sqrt(np.mean(errors ** 2))
        
        return H, rmse
    
    def save(self, output_path: str, H: np.ndarray, rmse: float):
        """
        Save calibration to JSON file.
        
        Args:
            output_path: Output JSON file path
            H: Homography matrix (3x3)
            rmse: Root mean square error (meters)
        """
        calib_data = {
            'H': H.tolist(),
            'rmse': float(rmse),
            'image_points': self.image_points,
            'world_points': self.world_points,
            'image_path': str(self.image_path)
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(calib_data, f, indent=2)
        
        print(f"\nCalibration saved to: {output_path}")
        print(f"RMSE: {rmse:.3f} meters")
        if rmse > 1.5:
            print("Warning: RMSE > 1.5m, consider recalibrating with more points")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Homography Calibration Tool')
    parser.add_argument('--image', required=True, help='Path to calibration image')
    parser.add_argument('--save', required=True, help='Output JSON file path (e.g., config/calib/camera-id_h.json)')
    
    args = parser.parse_args()
    
    try:
        calibrator = HomographyCalibrator(args.image)
        H, rmse = calibrator.calibrate()
        calibrator.save(args.save, H, rmse)
        
        print("\nCalibration complete!")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

