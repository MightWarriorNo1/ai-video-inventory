// Trailer Vision Edge Dashboard JavaScript

const METRICS_URL = '/metrics.json';
const EVENTS_URL = '/events';
const REFRESH_INTERVAL = 2000; // 2 seconds

let metricsInterval;
let eventsInterval;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    loadMetrics();
    loadEvents();
    setupCameraFilter();
    
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
        tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: #6b7280; padding: 20px;">No events found</td></tr>';
        return;
    }
    
    // Reverse to show newest first
    events.reverse().forEach(event => {
        const row = document.createElement('tr');
        
        const timestamp = event.ts_iso 
            ? new Date(event.ts_iso).toLocaleString() 
            : 'N/A';
        const confidence = parseFloat(event.conf) || 0;
        const confClass = confidence >= 0.7 ? 'confidence-high' 
                         : confidence >= 0.4 ? 'confidence-medium' 
                         : 'confidence-low';
        
        row.innerHTML = `
            <td>${timestamp}</td>
            <td>${event.camera_id || 'N/A'}</td>
            <td>${event.track_id || 'N/A'}</td>
            <td>${event.text || '-'}</td>
            <td><span class="confidence-badge ${confClass}">${(confidence * 100).toFixed(1)}%</span></td>
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

// Manual refresh functions
window.loadEvents = loadEvents;


