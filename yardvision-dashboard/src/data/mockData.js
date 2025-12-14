// Mock data for all pages

export const dashboardData = {
  kpis: {
    trailersOnYard: { value: 214, change: '+7 since 6am', icon: '🚛' },
    newDetections24h: { value: 392, ocrAccuracy: '95.2%', icon: '📈' },
    anomalies: { value: 4, description: 'plate mismatch / double-park', icon: '⚠️' },
    camerasOnline: { value: 3, degraded: 1, icon: '📷' },
  },
  queueStatus: {
    ingestQ: 3,
    ocrQ: 0,
    pubQ: 1,
  },
  accuracyChart: [
    { time: '06:00', detection: 92, ocr: 95 },
    { time: '07:00', detection: 92.5, ocr: 95.2 },
    { time: '08:00', detection: 93, ocr: 95.5 },
    { time: '09:00', detection: 93.2, ocr: 95.8 },
    { time: '10:00', detection: 93.5, ocr: 96 },
    { time: '11:00', detection: 93.8, ocr: 96.2 },
    { time: '12:00', detection: 94, ocr: 96.5 },
    { time: '13:00', detection: 94.2, ocr: 96.8 },
    { time: '14:00', detection: 94.5, ocr: 97 },
  ],
  yardUtilization: [
    { lane: 'A', utilization: 80 },
    { lane: 'B', utilization: 60 },
    { lane: 'C', utilization: 70 },
    { lane: 'D', utilization: 55 },
    { lane: 'Dock', utilization: 50 },
  ],
  cameraHealth: [
    { name: 'Gate East', id: 'CAM-01', fps: 22, latency: 86, uptime: 99.8, status: 'online' },
    { name: 'Lane A Row 1', id: 'CAM-02', fps: 24, latency: 92, uptime: 98.5, status: 'online' },
    { name: 'Lane A Row 2', id: 'CAM-03', fps: 18, latency: 145, uptime: 85.2, status: 'degraded' },
    { name: 'Dock Door 12', id: 'CAM-04', fps: 20, latency: 98, uptime: 97.3, status: 'online' },
  ],
}

export const inventoryData = {
  trailers: [
    { id: 'TRL-001', plate: 'ABC123', spot: 'A-12', status: 'Parked', detectedAt: '2024-01-15 08:30', ocrConfidence: 0.95 },
    { id: 'TRL-002', plate: 'XYZ789', spot: 'B-05', status: 'Parked', detectedAt: '2024-01-15 09:15', ocrConfidence: 0.92 },
    { id: 'TRL-003', plate: 'DEF456', spot: 'C-08', status: 'Parked', detectedAt: '2024-01-15 10:00', ocrConfidence: 0.88 },
    { id: 'TRL-004', plate: 'GHI789', spot: 'A-15', status: 'Parked', detectedAt: '2024-01-15 11:20', ocrConfidence: 0.96 },
    { id: 'TRL-005', plate: 'JKL012', spot: 'D-03', status: 'Parked', detectedAt: '2024-01-15 12:45', ocrConfidence: 0.91 },
  ],
  stats: {
    total: 214,
    parked: 210,
    inTransit: 4,
    anomalies: 2,
  },
}

export const yardViewData = {
  spots: [
    { id: 'A-01', lane: 'A', row: 1, occupied: true, trailerId: 'TRL-001', plate: 'ABC123' },
    { id: 'A-02', lane: 'A', row: 1, occupied: false, trailerId: null, plate: null },
    { id: 'A-03', lane: 'A', row: 1, occupied: true, trailerId: 'TRL-004', plate: 'GHI789' },
    { id: 'B-01', lane: 'B', row: 1, occupied: true, trailerId: 'TRL-002', plate: 'XYZ789' },
    { id: 'C-01', lane: 'C', row: 1, occupied: true, trailerId: 'TRL-003', plate: 'DEF456' },
    { id: 'D-01', lane: 'D', row: 1, occupied: true, trailerId: 'TRL-005', plate: 'JKL012' },
  ],
  lanes: ['A', 'B', 'C', 'D', 'Dock'],
}

export const eventsData = {
  events: [
    { id: 1, timestamp: '2024-01-15 14:30:22', camera: 'CAM-01', trackId: 'T-123', plate: 'ABC123', spot: 'A-12', type: 'Detection', confidence: 0.95 },
    { id: 2, timestamp: '2024-01-15 14:29:15', camera: 'CAM-02', trackId: 'T-124', plate: 'XYZ789', spot: 'B-05', type: 'OCR', confidence: 0.92 },
    { id: 3, timestamp: '2024-01-15 14:28:08', camera: 'CAM-03', trackId: 'T-125', plate: 'DEF456', spot: 'C-08', type: 'Anomaly', confidence: 0.88 },
    { id: 4, timestamp: '2024-01-15 14:27:45', camera: 'CAM-01', trackId: 'T-126', plate: 'GHI789', spot: 'A-15', type: 'Detection', confidence: 0.96 },
    { id: 5, timestamp: '2024-01-15 14:26:30', camera: 'CAM-04', trackId: 'T-127', plate: 'JKL012', spot: 'D-03', type: 'OCR', confidence: 0.91 },
  ],
  filters: ['All events', 'Detections', 'OCR', 'Anomalies', 'Spot Changes'],
}

export const configurationData = {
  cameras: [
    { id: 'CAM-01', name: 'Gate East', rtspUrl: 'rtsp://192.168.1.100:554/stream', enabled: true },
    { id: 'CAM-02', name: 'Lane A Row 1', rtspUrl: 'rtsp://192.168.1.101:554/stream', enabled: true },
    { id: 'CAM-03', name: 'Lane A Row 2', rtspUrl: 'rtsp://192.168.1.102:554/stream', enabled: true },
    { id: 'CAM-04', name: 'Dock Door 12', rtspUrl: 'rtsp://192.168.1.103:554/stream', enabled: true },
  ],
  processing: {
    detectEveryN: 5,
    ocrConfidenceThreshold: 0.5,
    trackerThreshold: 0.2,
  },
  integrations: {
    mqtt: { enabled: true, broker: 'mqtt://localhost:1883' },
    kafka: { enabled: false, broker: 'localhost:9092' },
    azure: { enabled: false, connectionString: '' },
  },
}

export const reportsData = {
  daily: {
    date: '2024-01-15',
    totalDetections: 392,
    ocrAccuracy: 95.2,
    anomalies: 4,
    avgProcessingTime: 86,
  },
  weekly: {
    week: 'Week 2, Jan 2024',
    totalDetections: 2456,
    ocrAccuracy: 94.8,
    anomalies: 28,
    avgProcessingTime: 92,
  },
  monthly: {
    month: 'January 2024',
    totalDetections: 11234,
    ocrAccuracy: 94.5,
    anomalies: 156,
    avgProcessingTime: 88,
  },
}


