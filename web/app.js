// Trailer Vision Edge Dashboard JavaScript

const METRICS_URL = '/metrics.json';
const REFRESH_INTERVAL = 2000; // 2 seconds

let metricsInterval;

// Video processing state
let currentVideoId = null;
let processingInterval = null;
let resultsInterval = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    loadMetrics();
    setupTabs();
    setupVideoProcessing();
    setupVideoRecording();
    setupDebugControls();
    setupApplicationControls();
    
    // Auto-refresh
    metricsInterval = setInterval(loadMetrics, REFRESH_INTERVAL);
});

// ========== TAB NAVIGATION ==========

function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');
            
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            document.getElementById(`${targetTab}-tab`).classList.add('active');
        });
    });
}

// Load and display metrics
async function loadMetrics() {
    try {
        const response = await fetch(METRICS_URL);
        const data = await response.json();
        
        updateStatus(true);
    } catch (error) {
        console.error('Error loading metrics:', error);
        updateStatus(false);
    }
}

// Update connection status
function updateStatus(connected) {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    
    if (connected) {
        statusDot.classList.remove('disconnected');
        statusText.textContent = 'Connected';
    } else {
        statusDot.classList.add('disconnected');
        statusText.textContent = 'Disconnected';
    }
}


// Video processing functions
function setupVideoProcessing() {
    const fileInput = document.getElementById('videoFileInput');
    const selectBtn = document.getElementById('selectVideoBtn');
    const processBtn = document.getElementById('processVideoBtn');
    const stopBtn = document.getElementById('stopProcessingBtn');
    
    selectBtn.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        try {
            selectBtn.disabled = true;
            selectBtn.textContent = 'Uploading...';
            
            const formData = new FormData();
            formData.append('video', file);
            
            const response = await fetch('/api/upload-video', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Upload failed');
            }
            
            const data = await response.json();
            currentVideoId = data.video_id;
            
            selectBtn.disabled = false;
            selectBtn.textContent = 'Select Video File';
            processBtn.disabled = false;
            
            updateProcessingStatus('Video uploaded successfully', 'success');
        } catch (error) {
            console.error('Error uploading video:', error);
            updateProcessingStatus('Upload failed: ' + error.message, 'error');
            selectBtn.disabled = false;
            selectBtn.textContent = 'Select Video File';
        }
    });
    
    processBtn.addEventListener('click', async () => {
        if (!currentVideoId) return;
        
        try {
            processBtn.disabled = true;
            stopBtn.disabled = false;
            updateProcessingStatus('Processing video...', 'processing');
            
            // Get detection mode from dropdown
            const detectionMode = document.getElementById('detectionMode').value;
            
            const response = await fetch(`/api/process-video/${currentVideoId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    detect_every_n: 5,
                    detection_mode: detectionMode
                })
            });
            
            if (!response.ok) {
                throw new Error('Processing failed to start');
            }
            
            // Start displaying processed video
            startProcessedVideoStream();
            
            // Start polling for results
            startResultsPolling();
            
        } catch (error) {
            console.error('Error starting processing:', error);
            updateProcessingStatus('Processing failed: ' + error.message, 'error');
            processBtn.disabled = false;
            stopBtn.disabled = true;
        }
    });
    
    stopBtn.addEventListener('click', async () => {
        try {
            await fetch('/api/stop-processing', { method: 'POST' });
            stopProcessing();
        } catch (error) {
            console.error('Error stopping processing:', error);
        }
    });
}

function startProcessedVideoStream() {
    const container = document.getElementById('processedVideoContainer');
    // Clear any existing content
    container.innerHTML = '';
    
    // Create image element for MJPEG stream
    const img = document.createElement('img');
    img.id = 'processedVideoStream';
    img.className = 'processed-video-stream';
    img.src = '/api/processed-video-stream?' + new Date().getTime(); // Add timestamp to prevent caching
    img.alt = 'Processed video';
    
    // Handle errors
    img.onerror = function() {
        console.error('Error loading processed video stream');
        // Try to reload after a delay
        setTimeout(() => {
            if (this.src) {
                this.src = '/api/processed-video-stream?' + new Date().getTime();
            }
        }, 2000);
    };
    
    // Handle load
    img.onload = function() {
        console.log('Processed video stream loaded');
    };
    
    container.appendChild(img);
}

function startResultsPolling() {
    if (resultsInterval) {
        clearInterval(resultsInterval);
    }
    
    let consecutiveNoProcessing = 0;
    
    resultsInterval = setInterval(async () => {
        try {
            // Poll both processing results and status
            const [resultsResponse, statusResponse] = await Promise.all([
                fetch('/api/processing-results'),
                fetch('/api/processing-status')
            ]);
            
            if (resultsResponse.ok) {
                const results = await resultsResponse.json();
                updateProcessingResults(results);
                
                // Check if processing is complete
                if (!results.processing) {
                    consecutiveNoProcessing++;
                    if (consecutiveNoProcessing > 3) {
                        // Processing is done
                        document.getElementById('processVideoBtn').disabled = false;
                        document.getElementById('stopProcessingBtn').disabled = true;
                        // Keep polling but less frequently to show final results
                    }
                } else {
                    consecutiveNoProcessing = 0;
                }
            }
            
            // Update status from status endpoint
            if (statusResponse.ok) {
                const status = await statusResponse.json();
                if (status.message) {
                    // Map backend status to frontend status types
                    let statusType = 'processing';
                    if (status.status === 'completed') {
                        statusType = 'success';
                    } else if (status.status === 'error') {
                        statusType = 'error';
                    } else if (status.status === 'processing_video' || status.status === 'processing_ocr') {
                        statusType = 'processing';
                    } else if (status.status === 'idle') {
                        statusType = 'idle';
                    }
                    updateProcessingStatus(status.message, statusType);
                }
            }
        } catch (error) {
            console.error('Error fetching results:', error);
        }
    }, 1000); // Poll every second
}

function updateProcessingResults(results) {
    document.getElementById('framesProcessed').textContent = results.frames_processed || 0;
    document.getElementById('totalDetections').textContent = results.detections || 0;
    document.getElementById('totalTracks').textContent = results.tracks || 0;
    document.getElementById('totalOCR').textContent = results.ocr_results || 0;
}

function stopProcessing() {
    if (resultsInterval) {
        clearInterval(resultsInterval);
        resultsInterval = null;
    }
    
    document.getElementById('processVideoBtn').disabled = false;
    document.getElementById('stopProcessingBtn').disabled = true;
    updateProcessingStatus('Processing stopped', 'stopped');
}

function updateProcessingStatus(message, type) {
    const statusEl = document.getElementById('processingStatus');
    statusEl.textContent = message;
    statusEl.className = `processing-status status-${type}`;
}

// Video recording functions
function setupVideoRecording() {
    const startBtn = document.getElementById('startRecordingBtn');
    const stopBtn = document.getElementById('stopRecordingBtn');
    const statusEl = document.getElementById('recordingStatus');
    const infoEl = document.getElementById('recordingInfo');
    
    // Check recording status on load
    checkRecordingStatus();
    
    // Poll recording status every 2 seconds
    setInterval(checkRecordingStatus, 2000);
    
    startBtn.addEventListener('click', async () => {
        try {
            startBtn.disabled = true;
            statusEl.textContent = 'Starting...';
            
            const response = await fetch('/api/start-recording', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            
            const data = await response.json();
            
            if (data.success) {
                startBtn.disabled = true;
                stopBtn.disabled = false;
                statusEl.textContent = 'Recording';
                statusEl.className = 'recording-status recording';
                infoEl.style.display = 'block';
                if (data.camera_id) {
                    document.getElementById('recordingCameraId').textContent = data.camera_id;
                    // Start displaying camera preview
                    startCameraPreview(data.camera_id);
                }
                updateRecordingStatus('Recording started', 'success');
            } else {
                startBtn.disabled = false;
                statusEl.textContent = 'Not recording';
                statusEl.className = 'recording-status';
                updateRecordingStatus('Failed to start: ' + data.message, 'error');
            }
        } catch (error) {
            console.error('Error starting recording:', error);
            startBtn.disabled = false;
            statusEl.textContent = 'Not recording';
            statusEl.className = 'recording-status';
            updateRecordingStatus('Error: ' + error.message, 'error');
        }
    });
    
    stopBtn.addEventListener('click', async () => {
        try {
            stopBtn.disabled = true;
            statusEl.textContent = 'Stopping...';
            
            const response = await fetch('/api/stop-recording', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                startBtn.disabled = false;
                stopBtn.disabled = true;
                statusEl.textContent = 'Not recording';
                statusEl.className = 'recording-status';
                infoEl.style.display = 'none';
                // Stop camera preview
                stopCameraPreview();
                updateRecordingStatus('Recording stopped. Video: ' + (data.video_path || 'N/A'), 'success');
            } else {
                stopBtn.disabled = false;
                updateRecordingStatus('Failed to stop: ' + data.message, 'error');
            }
        } catch (error) {
            console.error('Error stopping recording:', error);
            stopBtn.disabled = false;
            updateRecordingStatus('Error: ' + error.message, 'error');
        }
    });
}

async function checkRecordingStatus() {
    try {
        const response = await fetch('/api/recording-status');
        const data = await response.json();
        
        const startBtn = document.getElementById('startRecordingBtn');
        const stopBtn = document.getElementById('stopRecordingBtn');
        const statusEl = document.getElementById('recordingStatus');
        
        if (data.recording) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            statusEl.textContent = 'Recording';
            statusEl.className = 'recording-status recording';
            document.getElementById('recordingInfo').style.display = 'block';
            // Start camera preview if camera_id is available
            if (data.camera_id) {
                document.getElementById('recordingCameraId').textContent = data.camera_id;
                startCameraPreview(data.camera_id);
            }
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            statusEl.textContent = 'Not recording';
            statusEl.className = 'recording-status';
            document.getElementById('recordingInfo').style.display = 'none';
            // Stop camera preview
            stopCameraPreview();
        }
        
        // Update GPS status
        if (data.gps_available !== undefined) {
            document.getElementById('gpsStatus').textContent = data.gps_available ? 'Available' : 'Not available';
        }
    } catch (error) {
        console.error('Error checking recording status:', error);
    }
}

function updateRecordingStatus(message, type) {
    const statusEl = document.getElementById('recordingStatus');
    // Status is already updated in checkRecordingStatus, but we can add a temporary message
    console.log(`Recording: ${message} (${type})`);
}

// Camera preview functions
function startCameraPreview(cameraId) {
    const container = document.getElementById('cameraPreviewContainer');
    const previewImg = document.getElementById('cameraPreview');
    
    if (!container || !previewImg) return;
    
    // Show the preview container
    container.style.display = 'block';
    
    // Set the stream source
    previewImg.src = `/stream/${cameraId}?${new Date().getTime()}`;
    
    // Handle errors
    previewImg.onerror = function() {
        console.error('Error loading camera preview stream');
        // Try to reload after a delay
        setTimeout(() => {
            if (this.src) {
                this.src = `/stream/${cameraId}?${new Date().getTime()}`;
            }
        }, 2000);
    };
    
    // Handle load
    previewImg.onload = function() {
        console.log('Camera preview stream loaded');
    };
}

function stopCameraPreview() {
    const container = document.getElementById('cameraPreviewContainer');
    const previewImg = document.getElementById('cameraPreview');
    
    if (!container || !previewImg) return;
    
    // Hide the preview container
    container.style.display = 'none';
    
    // Clear the stream source
    previewImg.src = '';
}

// ========== DEBUG CONTROLS ==========

function setupDebugControls() {
    // Debug: Start Auto Recording
    const debugStartAutoRecordingBtn = document.getElementById('debugStartAutoRecordingBtn');
    const debugStopAutoRecordingBtn = document.getElementById('debugStopAutoRecordingBtn');
    const debugAutoRecordingStatus = document.getElementById('debugAutoRecordingStatus');
    
    debugStartAutoRecordingBtn.addEventListener('click', async () => {
        try {
            debugStartAutoRecordingBtn.disabled = true;
            debugAutoRecordingStatus.textContent = 'Starting...';
            debugAutoRecordingStatus.className = 'debug-status';
            
            const response = await fetch('/api/debug/start-auto-recording', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            
            const data = await response.json();
            
            if (data.success) {
                debugAutoRecordingStatus.textContent = `✓ ${data.message}`;
                debugAutoRecordingStatus.className = 'debug-status success';
                debugStartAutoRecordingBtn.disabled = true;
                debugStopAutoRecordingBtn.disabled = false;
            } else {
                debugAutoRecordingStatus.textContent = `✗ Error: ${data.error || data.message}`;
                debugAutoRecordingStatus.className = 'debug-status error';
                debugStartAutoRecordingBtn.disabled = false;
            }
        } catch (error) {
            console.error('Error starting auto recording:', error);
            debugAutoRecordingStatus.textContent = `✗ Error: ${error.message}`;
            debugAutoRecordingStatus.className = 'debug-status error';
            debugStartAutoRecordingBtn.disabled = false;
        }
    });
    
    // Debug: Stop Auto Recording
    debugStopAutoRecordingBtn.addEventListener('click', async () => {
        try {
            debugStopAutoRecordingBtn.disabled = true;
            debugAutoRecordingStatus.textContent = 'Stopping...';
            debugAutoRecordingStatus.className = 'debug-status';
            
            const response = await fetch('/api/debug/stop-auto-recording', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            
            const data = await response.json();
            
            if (data.success) {
                debugAutoRecordingStatus.textContent = `✓ ${data.message}`;
                debugAutoRecordingStatus.className = 'debug-status success';
                debugStartAutoRecordingBtn.disabled = false;
                debugStartAutoRecordingBtn.classList.remove('active');
                debugStopAutoRecordingBtn.disabled = true;
            } else {
                debugAutoRecordingStatus.textContent = `✗ Error: ${data.error || data.message}`;
                debugAutoRecordingStatus.className = 'debug-status error';
                debugStopAutoRecordingBtn.disabled = false;
            }
        } catch (error) {
            console.error('Error stopping auto recording:', error);
            debugAutoRecordingStatus.textContent = `✗ Error: ${error.message}`;
            debugAutoRecordingStatus.className = 'debug-status error';
            debugStopAutoRecordingBtn.disabled = false;
        }
    });
    
    // Debug: Start Video Processing
    const debugStartVideoProcessingBtn = document.getElementById('debugStartVideoProcessingBtn');
    const debugStopVideoProcessingBtn = document.getElementById('debugStopVideoProcessingBtn');
    const debugVideoPathInput = document.getElementById('debugVideoPathInput');
    const debugGpsLogPathInput = document.getElementById('debugGpsLogPathInput');
    const debugVideoProcessingStatus = document.getElementById('debugVideoProcessingStatus');
    
    debugStartVideoProcessingBtn.addEventListener('click', async () => {
        // Debug button always processes ALL videos in out/recordings directory
        const videoPath = debugVideoPathInput.value.trim();
        const processAll = true; // Always process all videos for debug button
        
        try {
            debugStartVideoProcessingBtn.disabled = true;
            debugStartVideoProcessingBtn.classList.add('active');
            debugVideoProcessingStatus.textContent = 'Finding videos in out/recordings...';
            debugVideoProcessingStatus.className = 'debug-status';
            
            const requestData = {
                video_path: videoPath || null, // Optional: can specify single video, but process_all takes precedence
                process_all: processAll, // Always true for debug button
                gps_log_path: debugGpsLogPathInput.value.trim() || null,
                detect_every_n: 5,
                detection_mode: 'trailer'
            };
            
            const response = await fetch('/api/debug/start-video-processing', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            const data = await response.json();
            
            if (data.success) {
                const message = data.videos_queued 
                    ? `✓ ${data.message} (${data.videos_queued} video(s) queued)`
                    : `✓ ${data.message}`;
                debugVideoProcessingStatus.textContent = message;
                debugVideoProcessingStatus.className = 'debug-status success';
                debugStartVideoProcessingBtn.disabled = true;
                debugStartVideoProcessingBtn.classList.add('active');
                debugStopVideoProcessingBtn.disabled = false;
            } else {
                debugVideoProcessingStatus.textContent = `✗ Error: ${data.error || data.message}`;
                debugVideoProcessingStatus.className = 'debug-status error';
                debugStartVideoProcessingBtn.disabled = false;
                debugStartVideoProcessingBtn.classList.remove('active');
            }
        } catch (error) {
            console.error('Error starting video processing:', error);
            debugVideoProcessingStatus.textContent = `✗ Error: ${error.message}`;
            debugVideoProcessingStatus.className = 'debug-status error';
            debugStartVideoProcessingBtn.disabled = false;
            debugStartVideoProcessingBtn.classList.remove('active');
        }
    });
    
    // Debug: Stop Video Processing
    debugStopVideoProcessingBtn.addEventListener('click', async () => {
        try {
            debugStopVideoProcessingBtn.disabled = true;
            debugVideoProcessingStatus.textContent = 'Stopping...';
            debugVideoProcessingStatus.className = 'debug-status';
            
            const response = await fetch('/api/debug/stop-video-processing', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            
            const data = await response.json();
            
            if (data.success) {
                debugVideoProcessingStatus.textContent = `✓ ${data.message}`;
                debugVideoProcessingStatus.className = 'debug-status success';
                debugStartVideoProcessingBtn.disabled = false;
                debugStartVideoProcessingBtn.classList.remove('active');
                debugStopVideoProcessingBtn.disabled = true;
            } else {
                debugVideoProcessingStatus.textContent = `✗ Error: ${data.error || data.message}`;
                debugVideoProcessingStatus.className = 'debug-status error';
                debugStopVideoProcessingBtn.disabled = false;
            }
        } catch (error) {
            console.error('Error stopping video processing:', error);
            debugVideoProcessingStatus.textContent = `✗ Error: ${error.message}`;
            debugVideoProcessingStatus.className = 'debug-status error';
            debugStopVideoProcessingBtn.disabled = false;
        }
    });
    
    // Debug: Start OCR Processing
    const debugStartOCRProcessingBtn = document.getElementById('debugStartOCRProcessingBtn');
    const debugStopOCRProcessingBtn = document.getElementById('debugStopOCRProcessingBtn');
    const debugCropsDirInput = document.getElementById('debugCropsDirInput');
    const debugOCRProcessingStatus = document.getElementById('debugOCRProcessingStatus');
    
    debugStartOCRProcessingBtn.addEventListener('click', async () => {
        const cropsDir = debugCropsDirInput.value.trim();
        const processAll = !cropsDir; // Process all if directory is empty
        
        try {
            debugStartOCRProcessingBtn.disabled = true;
            debugStartOCRProcessingBtn.classList.add('active');
            debugOCRProcessingStatus.textContent = processAll ? 'Queueing all crops...' : 'Queueing...';
            debugOCRProcessingStatus.className = 'debug-status';
            
            const requestData = {
                crops_dir: cropsDir || null,
                process_all: processAll,
                camera_id: 'test-video'
            };
            
            const response = await fetch('/api/debug/start-ocr-processing', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            const data = await response.json();
            
            if (data.success) {
                const message = data.ocr_jobs_queued 
                    ? `✓ ${data.message} (${data.ocr_jobs_queued} job(s) queued)`
                    : `✓ ${data.message}`;
                debugOCRProcessingStatus.textContent = message;
                debugOCRProcessingStatus.className = 'debug-status success';
                debugStartOCRProcessingBtn.disabled = true;
                debugStartOCRProcessingBtn.classList.add('active');
                debugStopOCRProcessingBtn.disabled = false;
            } else {
                debugOCRProcessingStatus.textContent = `✗ Error: ${data.error || data.message}`;
                debugOCRProcessingStatus.className = 'debug-status error';
                debugStartOCRProcessingBtn.disabled = false;
                debugStartOCRProcessingBtn.classList.remove('active');
            }
        } catch (error) {
            console.error('Error starting OCR processing:', error);
            debugOCRProcessingStatus.textContent = `✗ Error: ${error.message}`;
            debugOCRProcessingStatus.className = 'debug-status error';
            debugStartOCRProcessingBtn.disabled = false;
            debugStartOCRProcessingBtn.classList.remove('active');
        }
    });
    
    // Debug: Stop OCR Processing
    debugStopOCRProcessingBtn.addEventListener('click', async () => {
        try {
            debugStopOCRProcessingBtn.disabled = true;
            debugOCRProcessingStatus.textContent = 'Stopping...';
            debugOCRProcessingStatus.className = 'debug-status';
            
            const response = await fetch('/api/debug/stop-ocr-processing', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            
            const data = await response.json();
            
            if (data.success) {
                debugOCRProcessingStatus.textContent = `✓ ${data.message}`;
                debugOCRProcessingStatus.className = 'debug-status success';
                debugStartOCRProcessingBtn.disabled = false;
                debugStartOCRProcessingBtn.classList.remove('active');
                debugStopOCRProcessingBtn.disabled = true;
            } else {
                debugOCRProcessingStatus.textContent = `✗ Error: ${data.error || data.message}`;
                debugOCRProcessingStatus.className = 'debug-status error';
                debugStopOCRProcessingBtn.disabled = false;
            }
        } catch (error) {
            console.error('Error stopping OCR processing:', error);
            debugOCRProcessingStatus.textContent = `✗ Error: ${error.message}`;
            debugOCRProcessingStatus.className = 'debug-status error';
            debugStopOCRProcessingBtn.disabled = false;
        }
    });
    
    // Debug: Refresh Queue Status
    const debugRefreshQueueStatusBtn = document.getElementById('debugRefreshQueueStatusBtn');
    const debugQueueStatus = document.getElementById('debugQueueStatus');
    
    async function refreshQueueStatus() {
        try {
            const response = await fetch('/api/debug/processing-queue-status');
            const data = await response.json();
            
            if (data.available && data.status) {
                const status = data.status;
                debugQueueStatus.innerHTML = `
                    <div class="status-item">
                        <strong>Processing Video:</strong> ${status.processing_video ? '✓ Yes' : '✗ No'}
                    </div>
                    <div class="status-item">
                        <strong>Processing OCR:</strong> ${status.processing_ocr ? '✓ Yes' : '✗ No'}
                    </div>
                    <div class="status-item">
                        <strong>Video Queue Size:</strong> ${status.video_queue_size}
                    </div>
                    <div class="status-item">
                        <strong>OCR Queue Size:</strong> ${status.ocr_queue_size}
                    </div>
                    <div class="status-item">
                        <strong>Videos Queued:</strong> ${status.stats.videos_queued}
                    </div>
                    <div class="status-item">
                        <strong>Videos Processed:</strong> ${status.stats.videos_processed}
                    </div>
                    <div class="status-item">
                        <strong>OCR Jobs Queued:</strong> ${status.stats.ocr_jobs_queued}
                    </div>
                    <div class="status-item">
                        <strong>OCR Jobs Processed:</strong> ${status.stats.ocr_jobs_processed}
                    </div>
                    <div class="status-item">
                        <strong>Errors:</strong> ${status.stats.errors}
                    </div>
                `;
            } else {
                debugQueueStatus.innerHTML = `<div class="status-item error">Processing queue not available</div>`;
            }
        } catch (error) {
            console.error('Error refreshing queue status:', error);
            debugQueueStatus.innerHTML = `<div class="status-item error">Error: ${error.message}</div>`;
        }
    }
    
    debugRefreshQueueStatusBtn.addEventListener('click', refreshQueueStatus);
    
    // Auto-refresh queue status every 3 seconds
    setInterval(refreshQueueStatus, 3000);
    
    // Initial load
    refreshQueueStatus();
    
    // Check initial status for stop buttons
    checkDebugProcessStatus();
    
    // Periodically check and update stop button states
    setInterval(() => {
        updateDebugButtonStates();
    }, 2000);
}

// Update debug button states based on current processing status
async function updateDebugButtonStates() {
    // Check recording status
    try {
        const response = await fetch('/api/recording-status');
        const data = await response.json();
        const debugStartAutoRecordingBtn = document.getElementById('debugStartAutoRecordingBtn');
        const debugStopAutoRecordingBtn = document.getElementById('debugStopAutoRecordingBtn');
        
        if (debugStartAutoRecordingBtn && debugStopAutoRecordingBtn) {
            if (data.recording) {
                debugStartAutoRecordingBtn.disabled = true;
                debugStartAutoRecordingBtn.classList.add('active');
                debugStopAutoRecordingBtn.disabled = false;
            } else {
                debugStartAutoRecordingBtn.disabled = false;
                debugStartAutoRecordingBtn.classList.remove('active');
                debugStopAutoRecordingBtn.disabled = true;
            }
        }
    } catch (error) {
        // Silently fail - don't spam console
    }
    
    // Check video processing and OCR status
    try {
        const response = await fetch('/api/debug/processing-queue-status');
        const data = await response.json();
        
        if (data.available && data.status) {
            const status = data.status;
            
            // Update video processing buttons
            const debugStartVideoProcessingBtn = document.getElementById('debugStartVideoProcessingBtn');
            const debugStopVideoProcessingBtn = document.getElementById('debugStopVideoProcessingBtn');
            
            if (debugStartVideoProcessingBtn && debugStopVideoProcessingBtn) {
                if (status.processing_video || status.video_queue_size > 0) {
                    debugStartVideoProcessingBtn.disabled = true;
                    debugStopVideoProcessingBtn.disabled = false;
                } else {
                    debugStartVideoProcessingBtn.disabled = false;
                    debugStopVideoProcessingBtn.disabled = true;
                }
            }
            
            // Update OCR processing buttons
            const debugStartOCRProcessingBtn = document.getElementById('debugStartOCRProcessingBtn');
            const debugStopOCRProcessingBtn = document.getElementById('debugStopOCRProcessingBtn');
            
            if (debugStartOCRProcessingBtn && debugStopOCRProcessingBtn) {
                if (status.processing_ocr || status.ocr_queue_size > 0) {
                    debugStartOCRProcessingBtn.disabled = true;
                    debugStopOCRProcessingBtn.disabled = false;
                } else {
                    debugStartOCRProcessingBtn.disabled = false;
                    debugStopOCRProcessingBtn.disabled = true;
                }
            }
        }
    } catch (error) {
        // Silently fail - don't spam console
    }
}

// Check initial status of debug processes
function checkDebugProcessStatus() {
    // Use the same update function
    updateDebugButtonStates();
}

// ========== APPLICATION CONTROLS ==========

function setupApplicationControls() {
    const startAppBtn = document.getElementById('startApplicationBtn');
    const stopAppBtn = document.getElementById('stopApplicationBtn');
    const appStatus = document.getElementById('applicationStatus');
    const appInfo = document.getElementById('applicationInfo');
    const appStatusText = document.getElementById('appStatusText');
    const appCameraId = document.getElementById('appCameraId');
    const appRecordingStatus = document.getElementById('appRecordingStatus');
    const appVideoQueue = document.getElementById('appVideoQueue');
    const appOCRQueue = document.getElementById('appOCRQueue');
    
    // Start Application
    startAppBtn.addEventListener('click', async () => {
        try {
            startAppBtn.disabled = true;
            appStatus.textContent = 'Loading assets...';
            appStatus.className = 'app-status starting';
            
            const response = await fetch('/api/start-application', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            
            const data = await response.json();
            
            if (data.success) {
                appStatus.textContent = '✓ Running';
                appStatus.className = 'app-status running';
                startAppBtn.disabled = true;
                stopAppBtn.disabled = false;
                appInfo.style.display = 'block';
                appStatusText.textContent = 'Running';
                appCameraId.textContent = data.camera_id || '-';
                
                // Display assets loaded status
                if (data.assets_loaded) {
                    console.log('Assets loaded:', data.assets_loaded);
                }
                
                // Start status polling
                startApplicationStatusPolling();
            } else {
                appStatus.textContent = `✗ ${data.message || 'Failed to start'}`;
                appStatus.className = 'app-status error';
                startAppBtn.disabled = false;
                
                // Show assets status if available
                if (data.assets_loaded) {
                    console.error('Assets status:', data.assets_loaded);
                }
            }
        } catch (error) {
            console.error('Error starting application:', error);
            appStatus.textContent = `✗ Error: ${error.message}`;
            appStatus.className = 'app-status error';
            startAppBtn.disabled = false;
        }
    });
    
    // Stop Application
    stopAppBtn.addEventListener('click', async () => {
        try {
            stopAppBtn.disabled = true;
            appStatus.textContent = 'Stopping recording...';
            appStatus.className = 'app-status stopping';
            
            const response = await fetch('/api/stop-application', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Check if processing is still ongoing
                if (data.result && data.result.processing_ongoing) {
                    appStatus.textContent = 'Application stopped, but processing is still on. Please wait until the process is completed';
                    appStatus.className = 'app-status stopping';
                    stopAppBtn.disabled = true; // Keep disabled during graceful shutdown
                    appInfo.style.display = 'block';
                    // Continue polling to show processing status
                    startApplicationStatusPolling();
                } else {
                    appStatus.textContent = 'Stopped';
                    appStatus.className = 'app-status stopped';
                    startAppBtn.disabled = false;
                    stopAppBtn.disabled = true;
                    appInfo.style.display = 'none';
                    
                    // Stop status polling
                    if (appStatusPollingInterval) {
                        clearInterval(appStatusPollingInterval);
                        appStatusPollingInterval = null;
                    }
                }
            } else {
                appStatus.textContent = `✗ ${data.message || 'Failed to stop'}`;
                appStatus.className = 'app-status error';
                stopAppBtn.disabled = false;
            }
        } catch (error) {
            console.error('Error stopping application:', error);
            appStatus.textContent = `✗ Error: ${error.message}`;
            appStatus.className = 'app-status error';
            stopAppBtn.disabled = false;
        }
    });
    
    // Application status polling
    let appStatusPollingInterval = null;
    
    function startApplicationStatusPolling() {
        if (appStatusPollingInterval) {
            clearInterval(appStatusPollingInterval);
        }
        
        async function pollStatus() {
            try {
                const response = await fetch('/api/application-status');
                const data = await response.json();
                
                // Check if gracefully shutting down
                if (data.gracefully_shutting_down) {
                    appStatus.textContent = 'Application stopped, but processing is still on. Please wait until the process is completed';
                    appStatus.className = 'app-status stopping';
                    stopAppBtn.disabled = true;
                    appInfo.style.display = 'block';
                    appStatusText.textContent = 'Processing...';
                    appRecordingStatus.textContent = '✗ Not Recording';
                } else if (data.running) {
                    appRecordingStatus.textContent = '✓ Recording';
                    appStatusText.textContent = 'Running';
                } else {
                    appRecordingStatus.textContent = '✗ Not Recording';
                    appStatusText.textContent = 'Stopped';
                    // If not processing and not recording, fully stopped
                    if (!data.processing_ongoing) {
                        appStatus.textContent = 'Stopped';
                        appStatus.className = 'app-status stopped';
                        startAppBtn.disabled = false;
                        stopAppBtn.disabled = true;
                        if (appStatusPollingInterval) {
                            clearInterval(appStatusPollingInterval);
                            appStatusPollingInterval = null;
                        }
                    }
                }
                
                if (data.queue_status) {
                    appVideoQueue.textContent = `${data.queue_status.video_queue_size} jobs`;
                    appOCRQueue.textContent = `${data.queue_status.ocr_queue_size} jobs`;
                } else {
                    appVideoQueue.textContent = 'N/A';
                    appOCRQueue.textContent = 'N/A';
                }
            } catch (error) {
                console.error('Error polling application status:', error);
            }
        }
        
        // Poll immediately and then every 2 seconds
        pollStatus();
        appStatusPollingInterval = setInterval(pollStatus, 2000);
    }
    
    // Check initial status on load
    async function checkInitialStatus() {
        try {
            const response = await fetch('/api/application-status');
            const data = await response.json();
            
            if (data.running) {
                appStatus.textContent = '✓ Running';
                appStatus.className = 'app-status running';
                startAppBtn.disabled = true;
                stopAppBtn.disabled = false;
                appInfo.style.display = 'block';
                startApplicationStatusPolling();
            } else {
                // Ensure Stop button is disabled when not running
                startAppBtn.disabled = false;
                stopAppBtn.disabled = true;
                appStatus.textContent = 'Ready';
                appStatus.className = 'app-status stopped';
            }
        } catch (error) {
            console.error('Error checking initial status:', error);
            // On error, ensure initial state
            startAppBtn.disabled = false;
            stopAppBtn.disabled = true;
            appStatus.textContent = 'Ready';
            appStatus.className = 'app-status stopped';
        }
    }
    
    checkInitialStatus();
}



