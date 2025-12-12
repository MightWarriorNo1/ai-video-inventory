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
            
            const response = await fetch(`/api/process-video/${currentVideoId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ detect_every_n: 5 })
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
            const response = await fetch('/api/processing-results');
            if (response.ok) {
                const results = await response.json();
                updateProcessingResults(results);
                
                // Check if processing is complete
                if (!results.processing) {
                    consecutiveNoProcessing++;
                    if (consecutiveNoProcessing > 3) {
                        // Processing is done
                        updateProcessingStatus('Processing complete', 'success');
                        document.getElementById('processVideoBtn').disabled = false;
                        document.getElementById('stopProcessingBtn').disabled = true;
                        // Keep polling but less frequently to show final results
                    }
                } else {
                    consecutiveNoProcessing = 0;
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



