import { useState, useEffect } from 'react'
import { fetchCameras } from '../services/api'
import CameraHealthTable from '../components/CameraHealthTable'
import './CameraHealth.css'

const CameraHealth = () => {
  const [cameras, setCameras] = useState([])
  const [loading, setLoading] = useState(true)

  const loadCameras = async (forceRefresh = false) => {
    setLoading(true)
    const camerasData = await fetchCameras(forceRefresh)
    setCameras(camerasData)
    setLoading(false)
  }

  useEffect(() => {
    loadCameras()
    
    // Listen for refresh event from sync button
    const handleRefresh = () => {
      loadCameras(true)  // Force refresh when sync button is clicked
    }
    window.addEventListener('refresh-data', handleRefresh)
    
    return () => {
      window.removeEventListener('refresh-data', handleRefresh)
    }
  }, [])

  if (loading) {
    return (
      <div className="camera-health-page">
        <div className="loading-message">Loading camera data...</div>
      </div>
    )
  }

  const onlineCount = cameras.filter(c => c.status === 'online').length
  const degradedCount = cameras.filter(c => c.status === 'degraded').length
  const offlineCount = cameras.filter(c => c.status === 'offline').length
  const avgFps = cameras.length > 0 
    ? Math.round(cameras.reduce((sum, c) => sum + c.fps, 0) / cameras.length) 
    : 0
  const avgLatency = cameras.length > 0
    ? Math.round(cameras.reduce((sum, c) => sum + c.latency, 0) / cameras.length)
    : 0

  return (
    <div className="camera-health-page">
      <div className="camera-stats">
        <div className="stat-card">
          <div className="stat-label">Total Cameras</div>
          <div className="stat-value">{cameras.length}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Online</div>
          <div className="stat-value" style={{ color: '#10b981' }}>{onlineCount}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Degraded</div>
          <div className="stat-value" style={{ color: '#f59e0b' }}>{degradedCount}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Offline</div>
          <div className="stat-value" style={{ color: '#ef4444' }}>{offlineCount}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg FPS</div>
          <div className="stat-value">{avgFps}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg Latency</div>
          <div className="stat-value">{avgLatency}ms</div>
        </div>
      </div>

      <div className="camera-health-table-container">
        <h3>Camera Health Status</h3>
        {cameras.length === 0 ? (
          <div className="no-cameras-message">
            No cameras connected. Make sure your video processing pipeline is running.
          </div>
        ) : (
          <CameraHealthTable cameras={cameras} />
        )}
      </div>
    </div>
  )
}

export default CameraHealth









