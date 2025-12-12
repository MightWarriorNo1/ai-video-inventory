#!/usr/bin/env python3
"""
Test YOLOv8 Detection and ByteTrack Tracking on Single Image

This script tests YOLOv8 detection and ByteTrack tracking on a single image.
It visualizes the detection results with bounding boxes and track IDs.

Supports both:
- Pre-trained COCO models (default: yolov8m.pt, detects trucks - class 7)
- Fine-tuned models (e.g., runs/detect/trailer_back_detector/weights/best.pt, detects trailers - class 0)

Usage:
    python test/test_yolo_bytetrack.py <image_path> [--output <output_path>] [--conf <confidence>] [--model <model_path>] [--target-class <class_id>]
    
Examples:
    # Pre-trained COCO model (trucks)
    python test/test_yolo_bytetrack.py img_00675.jpg --output result.jpg --conf 0.25
    
    # Fine-tuned model (trailers)
    python test/test_yolo_bytetrack.py img_00675.jpg --model runs/detect/trailer_back_detector/weights/best.pt --target-class 0
"""

import sys
import os
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.ai.detector_yolov8 import YOLOv8Detector
from app.ai.tracker_bytetrack import ByteTrackWrapper
from app.image_preprocessing import ImagePreprocessor


def draw_detections(image, detections, color=(0, 255, 0), label_prefix="Det"):
    """Draw detection bounding boxes on image."""
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        conf = det.get('conf', 0.0)
        cls = det.get('cls', 0)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{label_prefix} cls:{cls} conf:{conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - baseline - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image


def compute_iou(bbox1, bbox2):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def draw_tracks(image, tracks, color=(255, 0, 0)):
    """Draw tracked bounding boxes with track IDs on image."""
    for track in tracks:
        bbox = track['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        track_id = track.get('track_id', 0)
        conf = track.get('conf', 0.0)
        cls = track.get('cls', 0)
        
        # Draw bounding box (thicker for tracks)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # Draw track ID label
        label = f"Track ID:{track_id} cls:{cls} conf:{conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - baseline - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image


def test_yolo_bytetrack(image_path, output_path=None, conf_threshold=0.20, use_preprocessing=True, 
                        show_all_detections=False, model_path=None, target_class=None):
    """
    Test YOLO detection and ByteTrack tracking on a single image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image (optional)
        conf_threshold: Confidence threshold for detections
        use_preprocessing: Whether to use image preprocessing
    """
    print("=" * 60)
    print("YOLO + ByteTrack Test on Single Image")
    print("=" * 60)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"✗ Error: Image not found: {image_path}")
        return False
    
    # Load image
    print(f"\n[1/5] Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"✗ Error: Failed to load image: {image_path}")
        return False
    
    print(f"  ✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Initialize preprocessor (optional)
    preprocessed_image = image
    if use_preprocessing:
        print(f"\n[2/5] Initializing image preprocessor...")
        preprocessor = ImagePreprocessor(
            enable_yolo_preprocessing=True,
            yolo_strategy="enhanced"
        )
        preprocessed_image = preprocessor.preprocess_for_yolo(image)
        print(f"  ✓ Image preprocessed for YOLO")
    else:
        print(f"\n[2/5] Skipping image preprocessing")
    
    # Initialize YOLO detector
    print(f"\n[3/5] Initializing YOLOv8 detector...")
    try:
        # Determine model path and target class
        if model_path is None:
            # Default: use pre-trained COCO model
            model_path = "yolov8m.pt"
            default_target_class = 7  # COCO class 7 = truck
            model_description = "yolov8m.pt (COCO pre-trained)"
        else:
            # Use fine-tuned model
            default_target_class = 0  # Fine-tuned models typically use class 0
            model_description = model_path
        
        # Use provided target_class or default
        if target_class is None:
            target_class = default_target_class
        
        detector = YOLOv8Detector(
            model_name=model_path,
            conf_threshold=conf_threshold,
            target_class=target_class,
            device=None  # Auto-detect device (CUDA if available, else CPU)
        )
        
        # Get actual class name from detector (reads from model)
        actual_class_name = "unknown"
        if hasattr(detector, 'class_names'):
            actual_class_name = detector.class_names.get(target_class, f'class_{target_class}')
        
        print(f"  ✓ YOLOv8 detector initialized")
        print(f"    - Model: {model_description}")
        print(f"    - Target class: {target_class} ({actual_class_name})")
        print(f"    - Confidence threshold: {conf_threshold}")
    except Exception as e:
        print(f"✗ Error initializing detector: {e}")
        print(f"\n  Troubleshooting tips:")
        print(f"    1. Ensure ultralytics is installed: pip install ultralytics")
        print(f"    2. The model will download automatically on first use")
        import traceback
        traceback.print_exc()
        return False
    
    # Run YOLO detection
    print(f"\n[4/5] Running YOLOv8 detection...")
    try:
        # Try detection with preprocessing
        detections = detector.detect(preprocessed_image)
        print(f"  ✓ Detection completed")
        print(f"    - Found {len(detections)} detections (confidence >= {conf_threshold})")
        
        # Analyze detection quality
        if len(detections) > 0:
            print(f"\n  [Detection Analysis]")
            for i, det in enumerate(detections):
                bbox = det['bbox']
                conf = det.get('conf', 0.0)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                print(f"    Detection {i+1}: conf={conf:.3f}, size={width:.0f}x{height:.0f} ({area:.0f} px²)")
            
            # Check for low confidence detections
            low_conf = [d for d in detections if d['conf'] < 0.3]
            if low_conf:
                print(f"\n  ⚠ WARNING: {len(low_conf)} detection(s) with low confidence (< 0.3):")
                for det in low_conf:
                    bbox = det['bbox']
                    conf = det.get('conf', 0.0)
                    print(f"    - conf={conf:.3f}, bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
                print(f"\n    Possible reasons:")
                print(f"    1. Partial occlusion - trailer partially hidden by others")
                print(f"    2. NMS suppression - better detection may have been removed")
                print(f"    3. Preprocessing issues - try --no-preprocessing")
                print(f"    4. Model limitation - model may not detect partially visible trailers well")
            
            # Check IoU between detections to see if NMS might be an issue
            if len(detections) >= 2:
                print(f"\n  [IoU Analysis] Checking overlap between detections...")
                for i in range(len(detections)):
                    for j in range(i+1, len(detections)):
                        iou = compute_iou(detections[i]['bbox'], detections[j]['bbox'])
                        if iou > 0.1:  # Only show if there's significant overlap
                            print(f"    Detection {i+1} vs {j+1}: IoU = {iou:.3f} (NMS threshold: 0.45)")
                            if iou > 0.45:
                                print(f"      ⚠ High overlap! NMS may have suppressed better detection")
        
        # Try without preprocessing if low confidence detected
        if len(detections) > 0 and any(d['conf'] < 0.3 for d in detections) and use_preprocessing:
            print(f"\n  [Testing without preprocessing]...")
            try:
                detections_no_prep = detector.detect(image)  # Use original image
                low_conf_count = len([d for d in detections if d['conf'] < 0.3])
                low_conf_no_prep = [d for d in detections_no_prep if d['conf'] < 0.3]
                
                # Check if no-preprocessing gives better results
                better_without_prep = False
                if len(low_conf_no_prep) < low_conf_count:
                    better_without_prep = True
                    print(f"    ✓ Better results without preprocessing!")
                    print(f"    - Low confidence detections: {len(low_conf_no_prep)} (was {low_conf_count})")
                    print(f"    → Using results WITHOUT preprocessing for better accuracy")
                elif len(detections_no_prep) > len(detections):
                    better_without_prep = True
                    print(f"    ✓ Found more detections without preprocessing: {len(detections_no_prep)} vs {len(detections)}")
                    print(f"    → Using results WITHOUT preprocessing")
                
                if better_without_prep:
                    # Replace detections with better results
                    detections = detections_no_prep
                    print(f"\n  [Updated Results] Using detections without preprocessing:")
                    for i, det in enumerate(detections):
                        bbox = det['bbox']
                        conf = det.get('conf', 0.0)
                        cls = det.get('cls', 0)
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        print(f"    Detection {i+1}: class={cls}, confidence={conf:.3f}, size={width:.0f}x{height:.0f}")
                else:
                    print(f"    → Preprocessing results are similar or better")
            except Exception as e:
                print(f"    ⚠ Could not test without preprocessing: {e}")
        
        # If show_all_detections is enabled, get raw detections with lower threshold
        all_detections = []
        if show_all_detections:
            print(f"\n  [Debug Mode] Checking for detections below threshold...")
            # Create a temporary detector with lower threshold to see all detections
            try:
                # Use same model and target class as main detector
                low_conf_detector = YOLOv8Detector(
                    model_name=model_path if model_path else "yolov8m.pt",
                    conf_threshold=0.05,  # Very low threshold to see all detections
                    target_class=target_class,
                    device=None
                )
                all_detections = low_conf_detector.detect(preprocessed_image)
                print(f"    - Found {len(all_detections)} total detections (confidence >= 0.05)")
                
                # Show detections below current threshold
                below_threshold = [d for d in all_detections if d['conf'] < conf_threshold]
                if below_threshold:
                    print(f"    - {len(below_threshold)} detections below threshold {conf_threshold}:")
                    for i, det in enumerate(sorted(below_threshold, key=lambda x: x['conf'], reverse=True)[:10]):
                        bbox = det['bbox']
                        conf = det.get('conf', 0.0)
                        cls = det.get('cls', 0)
                        print(f"      Below threshold {i+1}: class={cls}, confidence={conf:.3f}, bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
            except Exception as e:
                print(f"    ⚠ Could not get all detections: {e}")
        
        if len(detections) == 0:
            # Get class name for better error message
            class_name = "target objects"
            if hasattr(detector, 'class_names'):
                class_name = detector.class_names.get(target_class, f'class {target_class}')
            
            print(f"\n  ⚠ WARNING: No detections found!")
            print(f"    This could be due to:")
            print(f"    1. Confidence threshold too high (current: {conf_threshold})")
            print(f"    2. No {class_name} in image (detecting class {target_class})")
            print(f"    3. Model loading/inference error (check error messages above)")
            print(f"\n    Try running with:")
            print(f"    - Lower confidence: python test/test_yolo_bytetrack.py {image_path} --conf 0.10")
            print(f"    - Debug mode: python test/test_yolo_bytetrack.py {image_path} --show-all")
            if model_path and model_path != "yolov8m.pt":
                print(f"    - Check if model path is correct: {model_path}")
                print(f"    - Verify target class is correct (model may use different class IDs)")
        
        # Print detection details
        for i, det in enumerate(detections):
            bbox = det['bbox']
            conf = det.get('conf', 0.0)
            cls = det.get('cls', 0)
            print(f"    Detection {i+1}: class={cls}, confidence={conf:.3f}, bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    except Exception as e:
        print(f"✗ Error during detection: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Initialize ByteTrack tracker
    print(f"\n[5/5] Initializing ByteTrack tracker...")
    try:
        tracker = ByteTrackWrapper(
            track_thresh=conf_threshold,
            track_buffer=30,
            match_thresh=0.8
        )
        print(f"  ✓ ByteTrack tracker initialized")
        print(f"    - Track threshold: {conf_threshold}")
        print(f"    - Track buffer: 30 frames")
        print(f"    - Match threshold: 0.8")
    except Exception as e:
        print(f"✗ Error initializing tracker: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Run ByteTrack tracking
    print(f"\n[6/6] Running ByteTrack tracking...")
    try:
        tracks = tracker.update(detections, image)
        print(f"  ✓ Tracking completed")
        print(f"    - Found {len(tracks)} active tracks")
        
        # Print track details
        for track in tracks:
            track_id = track.get('track_id', 0)
            bbox = track['bbox']
            conf = track.get('conf', 0.0)
            cls = track.get('cls', 0)
            print(f"    Track ID {track_id}: class={cls}, confidence={conf:.3f}, bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    except Exception as e:
        print(f"✗ Error during tracking: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create visualization
    print(f"\n[Visualization] Creating output image...")
    vis_image = image.copy()
    
    # Draw all detections (below threshold) in yellow if debug mode
    if show_all_detections and all_detections:
        below_threshold_dets = [d for d in all_detections if d['conf'] < conf_threshold]
        if below_threshold_dets:
            vis_image = draw_detections(vis_image, below_threshold_dets, color=(0, 255, 255), label_prefix="Low")
    
    # Draw detections in green
    if detections:
        vis_image = draw_detections(vis_image, detections, color=(0, 255, 0), label_prefix="Det")
    
    # Draw tracks in blue (on top)
    if tracks:
        vis_image = draw_tracks(vis_image, tracks, color=(255, 0, 0))
    
    # Add summary text
    summary_y = 30
    cv2.putText(vis_image, f"Detections: {len(detections)}", (10, summary_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_image, f"Tracks: {len(tracks)}", (10, summary_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Save or display result
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"  ✓ Output saved to: {output_path}")
    else:
        # Auto-generate output path
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_result{input_path.suffix}"
        cv2.imwrite(str(output_path), vis_image)
        print(f"  ✓ Output saved to: {output_path}")
    
    # Display result (optional - requires GUI)
    try:
        cv2.imshow("YOLO + ByteTrack Result", vis_image)
        print(f"\n  Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print(f"  (Display window not available - image saved to file)")
    
    print(f"\n{'=' * 60}")
    print(f"✓ TEST COMPLETED SUCCESSFULLY")
    print(f"{'=' * 60}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test YOLO detection and ByteTrack tracking on a single image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use pre-trained COCO model (default)
    python test/test_yolo_bytetrack.py img_00675.jpg
    python test/test_yolo_bytetrack.py img_00675.jpg --output result.jpg
    python test/test_yolo_bytetrack.py img_00675.jpg --conf 0.25
    
    # Use fine-tuned model
    python test/test_yolo_bytetrack.py img_00675.jpg --model runs/detect/trailer_back_detector/weights/best.pt --target-class 0
    python test/test_yolo_bytetrack.py img_00675.jpg --model runs/detect/trailer_back_detector/weights/best.pt --conf 0.25
    
    # Debug mode
    python test/test_yolo_bytetrack.py img_00675.jpg --show-all
    python test/test_yolo_bytetrack.py img_00675.jpg --conf 0.25 --no-preprocessing
        """
    )
    
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to input image file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save output image (default: <input>_result.jpg)'
    )
    
    parser.add_argument(
        '--conf', '-c',
        type=float,
        default=0.20,
        help='Confidence threshold for detections (default: 0.20)'
    )
    
    parser.add_argument(
        '--no-preprocessing',
        action='store_true',
        help='Disable image preprocessing'
    )
    
    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Show all detections including those below confidence threshold (debug mode)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to YOLOv8 model file (default: yolov8m.pt for COCO pre-trained). Use trained model path like: runs/detect/trailer_back_detector/weights/best.pt'
    )
    
    parser.add_argument(
        '--target-class',
        type=int,
        default=None,
        help='Target class ID to detect (default: 7 for COCO truck, 0 for fine-tuned trailer models)'
    )
    
    args = parser.parse_args()
    
    success = test_yolo_bytetrack(
        args.image_path,
        output_path=args.output,
        conf_threshold=args.conf,
        use_preprocessing=not args.no_preprocessing,
        show_all_detections=args.show_all,
        model_path=args.model,
        target_class=args.target_class
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

