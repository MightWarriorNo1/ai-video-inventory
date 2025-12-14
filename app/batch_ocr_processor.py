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
        self.is_easyocr = ocr is not None and 'EasyOCRRecognizer' in type(ocr).__name__
        
        # IMPORTANT: Disable rotation in preprocessor if OCR handles rotation internally
        # oLmOCR and EasyOCR both handle rotation automatically, so pre-rotating images
        # causes double rotation issues and can reduce accuracy
        if self.preprocessor and (self.is_olmocr or self.is_easyocr):
            if self.preprocessor.enable_rotation:
                print(f"[BatchOCRProcessor] Disabling rotation in preprocessor (OCR handles rotation internally)")
                self.preprocessor.enable_rotation = False
    
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
            
            # Log which crop is being processed
            print(f"\n[BatchOCRProcessor] Processing crop {i+1}/{len(crops_metadata)}: {crop_filename}")
            
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
                print(f"[BatchOCRProcessor] {crop_filename}: Image size {w}x{h}, is_full_image={is_full_image}")
                
                # OCR parameters matching test script for best results
                # For cropped regions (trailer IDs): filter to alphanumeric, length 2-12
                # For full images: return all text, try multiple preprocessing
                base_ocr_params = {
                    'min_text_length': 2,
                    'max_text_length': 50,  # Increased from 40 to accommodate longer trailer IDs with company names
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
                    
                    # Try OCR with GPU first
                    result = None
                    ocr_failed = False
                    try:
                        result = self.ocr.recognize(crop_image, **ocr_params)
                        # Check if result is empty (might indicate GPU memory failure)
                        if not result.get('text', '').strip() and result.get('conf', 0.0) == 0.0:
                            # Empty result might be due to GPU memory error - try with resized image
                            print(f"[BatchOCRProcessor] Empty OCR result for {crop_filename} (full image mode), trying with resized image...")
                            # Resize image to 50% to reduce memory usage
                            h, w = crop_image.shape[:2]
                            resized = cv2.resize(crop_image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                            result = self.ocr.recognize(resized, **ocr_params)
                            if not result.get('text', '').strip():
                                print(f"[BatchOCRProcessor] Still empty after resize for {crop_filename}")
                    except RuntimeError as e:
                        error_str = str(e).lower()
                        if any(keyword in error_str for keyword in ["gpu", "memory", "cuda", "nvmap", "nvidia"]):
                            print(f"[BatchOCRProcessor] GPU memory error for {crop_filename}, trying with resized image...")
                            ocr_failed = True
                            # Try with resized image (50% size)
                            try:
                                h, w = crop_image.shape[:2]
                                resized = cv2.resize(crop_image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                                result = self.ocr.recognize(resized, **ocr_params)
                                ocr_failed = False
                            except Exception as resize_error:
                                print(f"[BatchOCRProcessor] Resized image also failed: {resize_error}")
                                result = {'text': '', 'conf': 0.0}
                    except Exception as e:
                        print(f"[BatchOCRProcessor] OCR error for {crop_filename}: {e}")
                        result = {'text': '', 'conf': 0.0}
                        ocr_failed = True
                    
                    # Clean GPU memory after each crop (especially important for oLmOCR)
                    if self.is_olmocr:
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except:
                            pass
                    
                    final_text = result.get('text', '') if result else ''
                    final_conf = result.get('conf', 0.0) if result else 0.0
                    
                    if not final_text.strip():
                        print(f"[BatchOCRProcessor] Final result empty for {crop_filename} (full image mode)")
                    
                    ocr_results[crop_filename] = {
                        'text': final_text,
                        'conf': final_conf,
                        'method': 'original' if not ocr_failed else 'error_retry'
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
                    failed_reasons = []
                    for prep in preprocessed_crops:
                        try:
                            # Use same parameters as test script
                            result = self.ocr.recognize(prep['image'], **ocr_params)
                            
                            # Debug: log full result dictionary to diagnose issues
                            print(f"[BatchOCRProcessor] DEBUG {crop_filename}: {prep['method']} result dict: {result}")
                            
                            result_text = result.get('text', '').strip()
                            result_conf = result.get('conf', 0.0)
                            
                            # Debug: log raw result to diagnose filtering issues
                            if not result_text and result_conf > 0.0:
                                print(f"[BatchOCRProcessor] DEBUG {crop_filename}: {prep['method']} returned conf={result_conf:.2f} but empty text - check OCR filtering")
                            elif result_text:
                                print(f"[BatchOCRProcessor] DEBUG {crop_filename}: {prep['method']} raw result: '{result_text[:50]}' (conf={result_conf:.2f}, len={len(result_text)})")
                            
                            if result_text:
                                all_results.append({
                                    'text': result_text,
                                    'conf': result_conf,
                                    'method': prep['method']
                                })
                                print(f"[BatchOCRProcessor] {crop_filename}: {prep['method']} succeeded: '{result_text[:50]}' (conf={result_conf:.2f})")
                            else:
                                # Result was empty - could be filtered or OCR failed
                                # Check if OCR actually ran but returned empty (might indicate filtering)
                                if result_conf == 0.0:
                                    failed_reasons.append(f"{prep['method']}: empty result (conf={result_conf:.2f})")
                                else:
                                    # Non-zero confidence but empty text suggests filtering
                                    failed_reasons.append(f"{prep['method']}: text filtered out (conf={result_conf:.2f})")
                        except Exception as e:
                            error_msg = str(e)
                            failed_reasons.append(f"{prep['method']}: exception - {error_msg[:100]}")
                            print(f"[BatchOCRProcessor] {crop_filename}: OCR error with {prep['method']}: {e}")
                    
                    # Log summary of failures if no results
                    if not all_results and failed_reasons:
                        print(f"[BatchOCRProcessor] {crop_filename}: All preprocessing methods failed:")
                        for reason in failed_reasons[:5]:  # Show first 5 failures
                            print(f"  - {reason}")
                        if len(failed_reasons) > 5:
                            print(f"  ... and {len(failed_reasons) - 5} more failures")
                    
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
                        # No results from any preprocessing method - try OCR's internal preprocessing as fallback
                        print(f"[BatchOCRProcessor] No OCR results found for {crop_filename} after trying {len(preprocessed_crops)} preprocessing methods")
                        print(f"[BatchOCRProcessor] Trying OCR's internal preprocessing (like test_olmocr.py)...")
                        
                        # Try with OCR's internal preprocessing (similar to test_olmocr.py)
                        fallback_params = base_ocr_params.copy()
                        fallback_params['try_multiple_preprocessing'] = True  # Use OCR's internal preprocessing
                        try:
                            print(f"[BatchOCRProcessor] DEBUG: Fallback params: {fallback_params}")
                            fallback_result = self.ocr.recognize(crop_image, **fallback_params)
                            print(f"[BatchOCRProcessor] DEBUG: Fallback result dict: {fallback_result}")
                            fallback_text = fallback_result.get('text', '').strip()
                            fallback_conf = fallback_result.get('conf', 0.0)
                            
                            if fallback_text:
                                print(f"[BatchOCRProcessor] Fallback succeeded: '{fallback_text[:50]}' (conf={fallback_conf:.2f})")
                                ocr_results[crop_filename] = {
                                    'text': fallback_text,
                                    'conf': fallback_conf,
                                    'method': 'ocr_internal_preprocessing'
                                }
                            else:
                                print(f"[BatchOCRProcessor] Fallback also returned empty result (conf={fallback_conf:.2f})")
                                print(f"[BatchOCRProcessor] DEBUG: Full fallback result: {fallback_result}")
                                ocr_results[crop_filename] = {
                                    'text': '',
                                    'conf': 0.0,
                                    'method': 'none'
                                }
                        except Exception as e:
                            print(f"[BatchOCRProcessor] Fallback failed: {e}")
                            import traceback
                            traceback.print_exc()
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
                    try:
                        result = self.ocr.recognize(crop_image, **ocr_params)
                        if not result.get('text', '').strip():
                            print(f"[BatchOCRProcessor] Empty OCR result for {crop_filename} (direct OCR)")
                        ocr_results[crop_filename] = {
                            'text': result.get('text', ''),
                            'conf': result.get('conf', 0.0),
                            'method': 'original'
                        }
                    except Exception as e:
                        print(f"[BatchOCRProcessor] OCR error for {crop_filename} (direct OCR): {e}")
                        ocr_results[crop_filename] = {
                            'text': '',
                            'conf': 0.0,
                            'method': 'error'
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
        
        # Generate summary of failed crops
        failed_crops = []
        for crop_filename, result in ocr_results.items():
            if not result.get('text', '').strip():
                failed_crops.append({
                    'filename': crop_filename,
                    'method': result.get('method', 'unknown'),
                    'conf': result.get('conf', 0.0)
                })
        
        if failed_crops:
            print(f"\n[BatchOCRProcessor] Summary of failed crops ({len(failed_crops)}):")
            for failed in failed_crops:
                print(f"  - {failed['filename']}: method={failed['method']}, conf={failed['conf']}")
        
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



