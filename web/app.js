// Trailer Vision Edge Dashboard JavaScript

const METRICS_URL = '/metrics.json';
const EVENTS_URL = '/events';
const REFRESH_INTERVAL = 2000; // 2 seconds

let metricsInterval;
let eventsInterval;

// Video processing state
let currentVideoId = null;
let processingInterval = null;
let resultsInterval = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    loadMetrics();
    loadEvents();
    setupCameraFilter();
    setupVideoFeeds();
    setupVideoProcessing();
    
    // Auto-refresh
    metricsInterval = setInterval(loadMetrics, REFRESH_INTERVAL);
    eventsInterval = setInterval(loadEvents, REFRESH_INTERVAL * 2);
});

// Load and display metrics
async function loadMetrics() {
    try {
        const response = await fetch(METRICS_URL);
        const data = await response.json();
        
        updateStatus(true);
        updateCameraMetrics(data.cameras);
        
        // Update video feeds if cameras changed
        updateVideoFeeds(data.cameras);
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

// Update camera metrics display
function updateCameraMetrics(cameras) {
    const container = document.getElementById('cameraMetrics');
    container.innerHTML = '';
    
    if (!cameras || Object.keys(cameras).length === 0) {
        container.innerHTML = '<p style="color: #6b7280; padding: 20px;">No camera data available</p>';
        return;
    }
    
    for (const [cameraId, metrics] of Object.entries(cameras)) {
        const card = document.createElement('div');
        card.className = 'camera-card';
        
        const fps = metrics.fps_ema?.toFixed(2) || '0.00';
        const frames = metrics.frames_processed || 0;
        const queueDepth = metrics.queue_depth || 0;
        const lastPublish = metrics.last_publish 
            ? new Date(metrics.last_publish).toLocaleString() 
            : 'Never';
        
        card.innerHTML = `
            <h3>${cameraId}</h3>
            <div class="metric-row">
                <span class="metric-label">FPS (EMA):</span>
                <span class="metric-value fps-value">${fps}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Frames Processed:</span>
                <span class="metric-value">${frames.toLocaleString()}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Queue Depth:</span>
                <span class="metric-value">${queueDepth}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Last Publish:</span>
                <span class="metric-value">${lastPublish}</span>
            </div>
        `;
        
        container.appendChild(card);
    }
}

// Load and display events
async function loadEvents() {
    try {
        const cameraFilter = document.getElementById('cameraFilter').value;
        const limit = parseInt(document.getElementById('eventLimit').value) || 50;
        
        let url = `${EVENTS_URL}?limit=${limit}`;
        if (cameraFilter) {
            url += `&camera_id=${encodeURIComponent(cameraFilter)}`;
        }
        
        const response = await fetch(url);
        const data = await response.json();
        
        updateEventsTable(data.events || []);
    } catch (error) {
        console.error('Error loading events:', error);
    }
}

// Update events table
function updateEventsTable(events) {
    const tbody = document.getElementById('eventsBody');
    tbody.innerHTML = '';
    
    if (events.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: #6b7280; padding: 20px;">No events found</td></tr>';
        return;
    }
    
    // Reverse to show newest first
    events.reverse().forEach(event => {
        const row = document.createElement('tr');
        
        const timestamp = event.ts_iso 
            ? new Date(event.ts_iso).toLocaleString() 
            : 'N/A';
        const ocrConfidence = parseFloat(event.conf) || 0;
        const detConfidence = parseFloat(event.det_conf) || 0;
        const ocrConfClass = ocrConfidence >= 0.7 ? 'confidence-high' 
                         : ocrConfidence >= 0.4 ? 'confidence-medium' 
                         : 'confidence-low';
        const detConfClass = detConfidence >= 0.7 ? 'confidence-high' 
                         : detConfidence >= 0.4 ? 'confidence-medium' 
                         : 'confidence-low';
        const trailerSize = (event.trailer_width && event.trailer_height) 
            ? `${event.trailer_width}×${event.trailer_height}` 
            : '-';
        
        row.innerHTML = `
            <td>${timestamp}</td>
            <td>${event.camera_id || 'N/A'}</td>
            <td>${event.track_id || 'N/A'}</td>
            <td style="font-weight: 600; color: #10b981;">${trailerSize}</td>
            <td style="font-weight: 600; color: #dc2626;">${event.text || '-'}</td>
            <td><span class="confidence-badge ${ocrConfClass}">${(ocrConfidence * 100).toFixed(1)}%</span></td>
            <td><span class="confidence-badge ${detConfClass}">${(detConfidence * 100).toFixed(1)}%</span></td>
            <td>${event.spot || 'N/A'}</td>
            <td>${event.method || 'N/A'}</td>
        `;
        
        tbody.appendChild(row);
    });
}

// Setup camera filter dropdown
async function setupCameraFilter() {
    try {
        const response = await fetch(METRICS_URL);
        const data = await response.json();
        
        const select = document.getElementById('cameraFilter');
        const cameras = Object.keys(data.cameras || {});
        
        cameras.forEach(cameraId => {
            const option = document.createElement('option');
            option.value = cameraId;
            option.textContent = cameraId;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error setting up camera filter:', error);
    }
}

// Setup video feeds for cameras (initial setup)
async function setupVideoFeeds() {
    try {
        const response = await fetch(METRICS_URL);
        const data = await response.json();
        updateVideoFeeds(data.cameras);
    } catch (error) {
        console.error('Error setting up video feeds:', error);
    }
}

// Update video feeds based on available cameras
function updateVideoFeeds(cameras) {
    const videoContainer = document.getElementById('videoFeeds');
    if (!videoContainer) return;
    
    const camerasList = Object.keys(cameras || {});
    
    if (camerasList.length === 0) {
        videoContainer.innerHTML = '<p style="color: #6b7280; padding: 20px; text-align: center;">No cameras available</p>';
        return;
    }
    
    // Get existing camera IDs
    const existingCameras = new Set();
    videoContainer.querySelectorAll('.video-wrapper').forEach(wrapper => {
        const title = wrapper.querySelector('.video-title');
        if (title) {
            existingCameras.add(title.textContent);
        }
    });
    
    // Add new cameras
    camerasList.forEach(cameraId => {
        if (!existingCameras.has(cameraId)) {
            const videoWrapper = document.createElement('div');
            videoWrapper.className = 'video-wrapper';
            
            const videoTitle = document.createElement('h3');
            videoTitle.textContent = cameraId;
            videoTitle.className = 'video-title';
            
            const video = document.createElement('img');
            video.className = 'video-feed';
            video.src = `/stream/${encodeURIComponent(cameraId)}`;
            video.alt = `Live feed from ${cameraId}`;
            video.onerror = function() {
                this.style.display = 'none';
                const errorMsg = document.createElement('div');
                errorMsg.className = 'video-error';
                errorMsg.textContent = 'Stream unavailable';
                videoWrapper.appendChild(errorMsg);
            };
            
            videoWrapper.appendChild(videoTitle);
            videoWrapper.appendChild(video);
            videoContainer.appendChild(videoWrapper);
        }
    });
    
    // Remove cameras that no longer exist
    videoContainer.querySelectorAll('.video-wrapper').forEach(wrapper => {
        const title = wrapper.querySelector('.video-title');
        if (title && !camerasList.includes(title.textContent)) {
            wrapper.remove();
        }
    });
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
    
    // Update events list
    const eventsContainer = document.getElementById('processingEvents');
    if (results.events && results.events.length > 0) {
        eventsContainer.innerHTML = '';
        
        // Show last 20 events
        const recentEvents = results.events.slice(-20).reverse();
        recentEvents.forEach(event => {
            const eventDiv = document.createElement('div');
            eventDiv.className = 'event-item';
            
            const timestamp = event.ts_iso 
                ? new Date(event.ts_iso).toLocaleTimeString() 
                : `Frame ${event.frame || 'N/A'}`;
            
            const text = event.text || '-';
            const conf = event.conf ? (event.conf * 100).toFixed(1) + '%' : '-';
            const detConf = event.det_conf ? (event.det_conf * 100).toFixed(1) + '%' : '-';
            const spot = event.spot || 'unknown';
            const trailerSize = (event.trailer_width && event.trailer_height) 
                ? `${event.trailer_width}x${event.trailer_height}px` 
                : 'N/A';
            
            eventDiv.innerHTML = `
                <div class="event-time">${timestamp}</div>
                <div class="event-details">
                    <span class="event-track">Track ${event.track_id || 'N/A'}</span>
                    <span class="event-trailer-size">Size: ${trailerSize}</span>
                    <span class="event-detection">Detection: ${detConf}</span>
                    <span class="event-text">Text: ${text}</span>
                    <span class="event-conf">OCR Conf: ${conf}</span>
                    <span class="event-spot">Spot: ${spot}</span>
                </div>
            `;
            
            eventsContainer.appendChild(eventDiv);
        });
    } else {
        if (eventsContainer.querySelector('.no-events')) {
            // Keep the placeholder
        } else {
            eventsContainer.innerHTML = '<p class="no-events">No events yet. Start processing to see results.</p>';
        }
    }
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

// Manual refresh functions
window.loadEvents = loadEvents;


