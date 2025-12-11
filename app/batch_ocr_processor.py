"""
Batch OCR Processor for Two-Stage Processing

Stage 1: Video processing saves cropped trailers (fast, no OCR)
Stage 2: Batch OCR processing on saved crops (can be run separately)
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class BatchOCRProcessor:
    """
    Process saved crop images with OCR in batch mode.
    
    This allows OCR to run separately from video processing for better performance.
    """
    
    def __init__(self, ocr, preprocessor=None):
        """
        Initialize batch OCR processor.
        
        Args:
            ocr: OCR recognizer instance (e.g., OlmOCRRecognizer)
            preprocessor: Optional ImagePreprocessor for preprocessing
        """
        self.ocr = ocr
        self.preprocessor = preprocessor
        
        # Check if OCR is oLmOCR (needs GPU memory cleanup)
        self.is_olmocr = ocr is not None and 'OlmOCRRecognizer' in type(ocr).__name__
    
    def process_crops_directory(self, crops_dir: str) -> Dict[str, Dict]:
        """
        Process all crops in a directory.
        
        Args:
            crops_dir: Path to directory containing crops and crops_metadata.json
            
        Returns:
            Dictionary mapping crop_filename -> OCR result
        """
        crops_path = Path(crops_dir)
        metadata_path = crops_path / "crops_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            crops_metadata = json.load(f)
        
        print(f"[BatchOCRProcessor] Processing {len(crops_metadata)} crops from {crops_dir}")
        
        ocr_results = {}
        processed = 0
        failed = 0
        
        for i, crop_meta in enumerate(crops_metadata):
            crop_path = Path(crop_meta['crop_path'])
            crop_filename = crop_meta['crop_filename']
            
            if not crop_path.exists():
                print(f"[BatchOCRProcessor] Warning: Crop file not found: {crop_path}")
                failed += 1
                continue
            
            try:
                # Load crop image
                crop_image = cv2.imread(str(crop_path))
                if crop_image is None:
                    print(f"[BatchOCRProcessor] Warning: Failed to load image: {crop_path}")
                    failed += 1
                    continue
                
                # Detect if this is a full image (large) or cropped region (small)
                # Use same logic as test script for consistency
                h, w = crop_image.shape[:2]
                is_full_image = max(h, w) > 800  # If image is larger than 800px, treat as full image
                
                # OCR parameters matching test script for best results
                # For cropped regions (trailer IDs): filter to alphanumeric, length 2-12
                # For full images: return all text, try multiple preprocessing
                base_ocr_params = {
                    'min_text_length': 2,
                    'max_text_length': 12,
                    'allowlist': "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",  # Alphanumeric only for trailer IDs
                }
                
                if is_full_image:
                    # Full image mode: return all text
                    base_ocr_params.update({
                        'full_image_mode': True,
                        'allowlist': None  # Don't filter for full images
                    })
                
                # Run OCR
                # For full images, match test script exactly: use direct OCR with try_multiple_preprocessing=True
                # (test script doesn't use preprocessor for full images)
                if is_full_image:
                    # Full image: use direct OCR with try_multiple_preprocessing=True (matching test script)
                    ocr_params = base_ocr_params.copy()
                    ocr_params['try_multiple_preprocessing'] = True  # Critical for full images!
                    result = self.ocr.recognize(crop_image, **ocr_params)
                    ocr_results[crop_filename] = {
                        'text': result.get('text', ''),
                        'conf': result.get('conf', 0.0),
                        'method': 'original'
                    }
                elif self.preprocessor and self.preprocessor.enable_ocr_preprocessing:
                    # Cropped region: use preprocessor (multiple preprocessing strategies)
                    # When using preprocessor, don't use try_multiple_preprocessing
                    # (preprocessor already handles multiple strategies)
                    ocr_params = base_ocr_params.copy()
                    ocr_params['try_multiple_preprocessing'] = False
                    # Get preprocessed versions
                    preprocessed_crops = self.preprocessor.preprocess_for_ocr(crop_image)
                    
                    # Try OCR on each preprocessed version with proper parameters
                    all_results = []
                    for prep in preprocessed_crops:
                        try:
                            # Use same parameters as test script
                            result = self.ocr.recognize(prep['image'], **ocr_params)
                            if result.get('text', '').strip():
                                all_results.append({
                                    'text': result.get('text', ''),
                                    'conf': result.get('conf', 0.0),
                                    'method': prep['method']
                                })
                        except Exception as e:
                            print(f"[BatchOCRProcessor] OCR error with {prep['method']}: {e}")
                    
                    # Select best result
                    if all_results:
                        if self.preprocessor:
                            best_result = self.preprocessor.select_best_ocr_result(all_results)
                        else:
                            best_result = max(all_results, key=lambda x: x.get('conf', 0.0))
                        
                        ocr_results[crop_filename] = {
                            'text': best_result['text'],
                            'conf': best_result['conf'],
                            'method': best_result.get('method', 'unknown')
                        }
                    else:
                        ocr_results[crop_filename] = {
                            'text': '',
                            'conf': 0.0,
                            'method': 'none'
                        }
                else:
                    # Cropped region: direct OCR without preprocessing, but with proper parameters
                    # For direct OCR on cropped regions, enable try_multiple_preprocessing for better results
                    ocr_params = base_ocr_params.copy()
                    ocr_params['try_multiple_preprocessing'] = True
                    result = self.ocr.recognize(crop_image, **ocr_params)
                    ocr_results[crop_filename] = {
                        'text': result.get('text', ''),
                        'conf': result.get('conf', 0.0),
                        'method': 'original'
                    }
                
                processed += 1
                
                # Progress update
                if (i + 1) % 10 == 0:
                    print(f"[BatchOCRProcessor] Progress: {i + 1}/{len(crops_metadata)} crops processed")
                
                # Clean GPU memory periodically
                if self.is_olmocr and (i + 1) % 5 == 0:
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except:
                        pass
                        
            except Exception as e:
                print(f"[BatchOCRProcessor] Error processing {crop_filename}: {e}")
                failed += 1
                ocr_results[crop_filename] = {
                    'text': '',
                    'conf': 0.0,
                    'method': 'error'
                }
        
        print(f"[BatchOCRProcessor] Completed: {processed} processed, {failed} failed")
        
        # Save OCR results
        results_path = crops_path / "ocr_results.json"
        with open(results_path, 'w') as f:
            json.dump(ocr_results, f, indent=2)
        print(f"[BatchOCRProcessor] Saved OCR results to {results_path}")
        
        return ocr_results
    
    def match_ocr_to_detections(self, crops_dir: str, ocr_results: Dict[str, Dict]) -> List[Dict]:
        """
        Match OCR results back to detection metadata.
        
        Args:
            crops_dir: Path to crops directory
            ocr_results: OCR results dictionary from process_crops_directory()
            
        Returns:
            List of combined results: YOLO + ByteTrack + OCR
        """
        crops_path = Path(crops_dir)
        metadata_path = crops_path / "crops_metadata.json"
        
        with open(metadata_path, 'r') as f:
            crops_metadata = json.load(f)
        
        combined_results = []
        
        for crop_meta in crops_metadata:
            crop_filename = crop_meta['crop_filename']
            
            # Get OCR result
            ocr_result = ocr_results.get(crop_filename, {
                'text': '',
                'conf': 0.0,
                'method': 'not_found'
            })
            
            # Combine all information
            combined = {
                # YOLO detection info
                'frame_count': crop_meta['frame_count'],
                'track_id': crop_meta['track_id'],
                'bbox': crop_meta['bbox'],
                'bbox_original': crop_meta['bbox_original'],
                'det_conf': crop_meta['det_conf'],
                'width': crop_meta['width'],
                'height': crop_meta['height'],
                
                # ByteTrack info
                'track_key': crop_meta['track_key'],
                'bbox_hash': crop_meta['bbox_hash'],
                
                # OCR result
                'ocr_text': ocr_result['text'],
                'ocr_conf': ocr_result['conf'],
                'ocr_method': ocr_result['method'],
                
                # Additional info
                'spot': crop_meta['spot'],
                'world_coords': crop_meta['world_coords'],
                'camera_id': crop_meta['camera_id'],
                'timestamp': crop_meta['timestamp'],
                'crop_path': crop_meta['crop_path']
            }
            
            combined_results.append(combined)
        
        # Save combined results
        combined_path = crops_path / "combined_results.json"
        with open(combined_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        print(f"[BatchOCRProcessor] Saved combined results to {combined_path}")
        
        return combined_results


def process_video_crops(video_crops_dir: str, ocr, preprocessor=None):
    """
    Convenience function to process crops for a video.
    
    Args:
        video_crops_dir: Path to video's crop directory (e.g., "out/crops/camera-01/video_name")
        ocr: OCR recognizer instance
        preprocessor: Optional ImagePreprocessor
    """
    processor = BatchOCRProcessor(ocr, preprocessor)
    
    # Process all crops
    ocr_results = processor.process_crops_directory(video_crops_dir)
    
    # Match OCR results to detections
    combined_results = processor.match_ocr_to_detections(video_crops_dir, ocr_results)
    
    return combined_results



