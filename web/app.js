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
    setupVideoProcessing();
    setupVideoRecording();
    
    // Auto-refresh
    metricsInterval = setInterval(loadMetrics, REFRESH_INTERVAL);
});

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



