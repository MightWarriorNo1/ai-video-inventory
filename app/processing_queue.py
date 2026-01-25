"""
Processing Queue Manager

Manages sequential processing of video chunks and OCR to prevent GPU memory conflicts.
Ensures only one GPU operation runs at a time (video processing OR OCR, never both).

Priority System:
- OCR has PRIORITY: When OCR is processing, video processing waits in queue
- Video processing waits: Checks OCR status before acquiring GPU lock
- Sequential execution: Only one GPU operation at a time

Workflow:
1. Video chunk saved → Queue video processing job
2. Video processing waits if OCR is active
3. Video processing completes → Queue OCR job on saved crops
4. OCR processing (has priority, video waits if queued)
5. OCR completes → Ready for server upload (future extension)
"""

import threading
import queue
import time
from pathlib import Path
from typing import Optional, Dict, Callable, Any
from datetime import datetime
import gc

# Try to import torch for GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ProcessingQueueManager:
    """
    Manages sequential processing queue for video and OCR operations.
    Ensures GPU is only used by one operation at a time.
    
    Priority: OCR has priority over video processing.
    - When OCR is processing, video processing jobs wait in queue
    - Video processing only starts when OCR is not active
    - GPU lock ensures only one operation uses GPU at a time
    """
    
    def __init__(self, video_processor, ocr, preprocessor=None, on_video_complete: Optional[Callable] = None, 
                 on_ocr_complete: Optional[Callable] = None):
        """
        Initialize processing queue manager.
        
        Args:
            video_processor: VideoProcessor instance
            ocr: OCR recognizer instance (for batch processing)
            preprocessor: Optional ImagePreprocessor instance
            on_video_complete: Optional callback(video_path, crops_dir, results) when video processing completes
            on_ocr_complete: Optional callback(video_path, crops_dir, ocr_results) when OCR completes
        """
        self.video_processor = video_processor
        self.ocr = ocr
        self.preprocessor = preprocessor
        self.on_video_complete = on_video_complete
        self.on_ocr_complete = on_ocr_complete
        
        # Processing queues
        self.video_queue = queue.Queue()
        self.ocr_queue = queue.Queue()
        
        # GPU lock - ensures only one GPU operation at a time
        self.gpu_lock = threading.Lock()
        
        # Processing state (thread-safe flags)
        self.processing_video = False
        self.processing_ocr = False
        self.running = True
        
        # Lock for state flags (for thread-safe checking)
        self.state_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'videos_queued': 0,
            'videos_processed': 0,
            'ocr_jobs_queued': 0,
            'ocr_jobs_processed': 0,
            'errors': 0
        }
        
        # Start worker threads
        self.video_worker_thread = threading.Thread(target=self._video_worker, daemon=True)
        self.ocr_worker_thread = threading.Thread(target=self._ocr_worker, daemon=True)
        self.video_worker_thread.start()
        self.ocr_worker_thread.start()
        
        print(f"[ProcessingQueueManager] Initialized with GPU lock protection")
    
    def queue_video_processing(self, video_path: str, camera_id: str, gps_log_path: Optional[str] = None, 
                               detect_every_n: int = 5, detection_mode: str = 'trailer'):
        """
        Queue a video for processing.
        
        Args:
            video_path: Path to video file
            camera_id: Camera identifier
            gps_log_path: Optional path to GPS log file
            detect_every_n: Run detector every N frames
            detection_mode: 'trailer' or 'car'
        """
        job = {
            'video_path': video_path,
            'camera_id': camera_id,
            'gps_log_path': gps_log_path,
            'detect_every_n': detect_every_n,
            'detection_mode': detection_mode,
            'queued_at': datetime.now().isoformat()
        }
        
        self.video_queue.put(job)
        self.stats['videos_queued'] += 1
        print(f"[ProcessingQueueManager] Queued video processing: {Path(video_path).name} (queue size: {self.video_queue.qsize()})")
    
    def _video_worker(self):
        """
        Worker thread that processes videos sequentially.
        Waits for OCR to complete before processing videos (OCR has priority).
        """
        from app.batch_ocr_processor import BatchOCRProcessor
        
        while self.running:
            try:
                # Get next video job (blocking with timeout to allow checking self.running)
                try:
                    job = self.video_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                video_path = job['video_path']
                camera_id = job['camera_id']
                gps_log_path = job.get('gps_log_path')
                detect_every_n = job.get('detect_every_n', 5)
                detection_mode = job.get('detection_mode', 'trailer')
                
                # CRITICAL: Wait for OCR to complete before processing video
                # OCR has priority - video processing must wait until OCR is completely done
                wait_start = time.time()
                with self.state_lock:
                    ocr_active = self.processing_ocr
                
                # Wait loop: Keep waiting while OCR is active
                while ocr_active and self.running:
                    elapsed = time.time() - wait_start
                    if elapsed > 1.0:  # Log every second while waiting
                        print(f"[ProcessingQueueManager] Video processing queued, waiting for OCR to complete: {Path(video_path).name} (waited {elapsed:.1f}s)")
                        wait_start = time.time()  # Reset to avoid spam
                    time.sleep(0.1)  # Check every 100ms
                    
                    # Re-check OCR status
                    with self.state_lock:
                        ocr_active = self.processing_ocr
                
                if not self.running:
                    # Put job back in queue if shutting down
                    self.video_queue.put(job)
                    break
                
                # At this point, OCR should be done. Proceed with video processing.
                print(f"[ProcessingQueueManager] OCR complete, starting video processing: {Path(video_path).name}")
                
                # Acquire GPU lock for video processing
                # This ensures no OCR can start while we're processing video
                with self.gpu_lock:
                    # Double-check OCR didn't start (shouldn't happen, but safety check)
                    with self.state_lock:
                        if self.processing_ocr:
                            print(f"[ProcessingQueueManager] ERROR: OCR became active while acquiring GPU lock! This should not happen.")
                            # Put job back and wait again
                            self.video_queue.put(job)
                            continue
                        
                        # All clear - start video processing
                        self.processing_video = True
                    try:
                        # Update video processor GPS log path if provided
                        if gps_log_path and hasattr(self.video_processor, 'gps_log_path'):
                            self.video_processor.gps_log_path = gps_log_path
                        
                        # Process video (detection, tracking, save crops - NO OCR)
                        results = None
                        crops_dir = None
                        
                        # Process video and collect results
                        for frame_num, processed_frame, events in self.video_processor.process_video(
                            video_path, 
                            camera_id=camera_id,
                            detect_every_n=detect_every_n
                        ):
                            # Just consume the generator - processing happens here
                            pass
                        
                        # Get results after processing
                        if hasattr(self.video_processor, 'get_results'):
                            results = self.video_processor.get_results()
                        
                        # Get crops directory
                        if hasattr(self.video_processor, 'crops_dir') and self.video_processor.crops_dir:
                            crops_dir = str(self.video_processor.crops_dir)
                        
                        # Cleanup GPU memory after video processing
                        self._cleanup_gpu_memory()
                        
                        print(f"[ProcessingQueueManager] Video processing complete: {Path(video_path).name}")
                        print(f"  - Crops saved to: {crops_dir}")
                        
                        self.stats['videos_processed'] += 1
                        
                        # Call completion callback
                        if self.on_video_complete:
                            try:
                                self.on_video_complete(video_path, crops_dir, results)
                            except Exception as e:
                                print(f"[ProcessingQueueManager] Error in video completion callback: {e}")
                        
                        # Queue OCR job if crops were saved
                        if crops_dir and Path(crops_dir).exists():
                            self._queue_ocr_job(video_path, crops_dir, camera_id)
                        else:
                            print(f"[ProcessingQueueManager] No crops directory found, skipping OCR")
                    
                    except Exception as e:
                        print(f"[ProcessingQueueManager] Error processing video {video_path}: {e}")
                        import traceback
                        traceback.print_exc()
                        self.stats['errors'] += 1
                        self._cleanup_gpu_memory()
                    
                    finally:
                        with self.state_lock:
                            self.processing_video = False
                
                self.video_queue.task_done()
                
            except Exception as e:
                print(f"[ProcessingQueueManager] Error in video worker thread: {e}")
                import traceback
                traceback.print_exc()
                self.stats['errors'] += 1
                time.sleep(1)  # Prevent tight error loop
    
    def _queue_ocr_job(self, video_path: str, crops_dir: str, camera_id: str):
        """Queue OCR processing for a video's crops."""
        job = {
            'video_path': video_path,
            'crops_dir': crops_dir,
            'camera_id': camera_id,
            'queued_at': datetime.now().isoformat()
        }
        
        self.ocr_queue.put(job)
        self.stats['ocr_jobs_queued'] += 1
        print(f"[ProcessingQueueManager] Queued OCR processing: {Path(crops_dir).name} (queue size: {self.ocr_queue.qsize()})")
    
    def _ocr_worker(self):
        """
        Worker thread that processes OCR jobs sequentially.
        OCR has priority - video processing will wait for OCR to complete.
        """
        from app.batch_ocr_processor import BatchOCRProcessor
        
        while self.running:
            try:
                # Get next OCR job
                try:
                    job = self.ocr_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                video_path = job['video_path']
                crops_dir = job['crops_dir']
                camera_id = job['camera_id']
                
                # Wait for any ongoing video processing to complete
                # OCR has priority, but we should let current video processing finish gracefully
                wait_start = time.time()
                with self.state_lock:
                    video_active = self.processing_video
                
                while video_active and self.running:
                    elapsed = time.time() - wait_start
                    if elapsed > 1.0:  # Log every second while waiting
                        print(f"[ProcessingQueueManager] OCR queued, waiting for video processing to complete: {Path(crops_dir).name} (waited {elapsed:.1f}s)")
                        wait_start = time.time()
                    time.sleep(0.1)
                    
                    # Re-check video status
                    with self.state_lock:
                        video_active = self.processing_video
                
                if not self.running:
                    # Put job back in queue if shutting down
                    self.ocr_queue.put(job)
                    break
                
                print(f"[ProcessingQueueManager] Starting OCR processing: {Path(crops_dir).name}")
                
                # Acquire GPU lock for OCR processing (OCR has priority)
                with self.gpu_lock:
                    # Set OCR processing flag BEFORE starting (so video worker knows to wait)
                    with self.state_lock:
                        self.processing_ocr = True
                    try:
                        # Initialize batch OCR processor
                        batch_processor = BatchOCRProcessor(self.ocr, self.preprocessor)
                        
                        # Process all crops
                        ocr_results = batch_processor.process_crops_directory(crops_dir)
                        
                        # Match OCR results to detections
                        combined_results = batch_processor.match_ocr_to_detections(crops_dir, ocr_results)
                        
                        # Cleanup GPU memory after OCR
                        self._cleanup_gpu_memory()
                        
                        print(f"[ProcessingQueueManager] OCR processing complete: {Path(crops_dir).name}")
                        print(f"  - Processed {len(combined_results)} crops")
                        
                        self.stats['ocr_jobs_processed'] += 1
                        
                        # Call completion callback
                        if self.on_ocr_complete:
                            try:
                                self.on_ocr_complete(video_path, crops_dir, combined_results)
                            except Exception as e:
                                print(f"[ProcessingQueueManager] Error in OCR completion callback: {e}")
                    
                    except Exception as e:
                        print(f"[ProcessingQueueManager] Error processing OCR {crops_dir}: {e}")
                        import traceback
                        traceback.print_exc()
                        self.stats['errors'] += 1
                        self._cleanup_gpu_memory()
                    
                    finally:
                        with self.state_lock:
                            self.processing_ocr = False
                
                self.ocr_queue.task_done()
                
            except Exception as e:
                print(f"[ProcessingQueueManager] Error in OCR worker thread: {e}")
                import traceback
                traceback.print_exc()
                self.stats['errors'] += 1
                time.sleep(1)  # Prevent tight error loop
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory after operations."""
        if not TORCH_AVAILABLE:
            return
        
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        with self.state_lock:
            return {
                'processing_video': self.processing_video,
                'processing_ocr': self.processing_ocr,
                'video_queue_size': self.video_queue.qsize(),
                'ocr_queue_size': self.ocr_queue.qsize(),
                'stats': self.stats.copy()
            }
    
    def stop(self):
        """Stop the processing queue manager."""
        print(f"[ProcessingQueueManager] Stopping...")
        self.running = False
        
        # Wait for queues to empty (with timeout)
        timeout = 30
        start_time = time.time()
        
        with self.state_lock:
            processing_video = self.processing_video
            processing_ocr = self.processing_ocr
        
        while (self.video_queue.qsize() > 0 or self.ocr_queue.qsize() > 0 or 
               processing_video or processing_ocr):
            if time.time() - start_time > timeout:
                print(f"[ProcessingQueueManager] Timeout waiting for queues to empty")
                break
            time.sleep(0.5)
            
            with self.state_lock:
                processing_video = self.processing_video
                processing_ocr = self.processing_ocr
        
        print(f"[ProcessingQueueManager] Stopped. Final stats: {self.stats}")
