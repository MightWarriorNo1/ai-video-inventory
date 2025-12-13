import { useState, useEffect } from 'react'
import { Search, ChevronDown } from 'lucide-react'
import { fetchDashboardData, fetchCameras } from '../services/api'
import { dashboardData as fallbackData } from '../data/mockData'
import KPICard from '../components/KPICard'
import AccuracyChart from '../components/AccuracyChart'
import YardUtilizationChart from '../components/YardUtilizationChart'
import CameraHealthTable from '../components/CameraHealthTable'
import QueueStatus from '../components/QueueStatus'
import RecentTrailerEvents from '../components/RecentTrailerEvents'
import YardMapSpotResolver from '../components/YardMapSpotResolver'
import './Dashboard.css'

const Dashboard = () => {
  const [data, setData] = useState(fallbackData)
  const [cameras, setCameras] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedEvent, setSelectedEvent] = useState(null)

  // Load data function that can be called manually
  const loadData = async (forceRefresh = false) => {
    setLoading(true)
    const [dashboardData, camerasData] = await Promise.all([
      fetchDashboardData(),
      fetchCameras(forceRefresh)  // Force camera refresh when sync is clicked
    ])
    
    if (dashboardData) {
      setData(dashboardData)
    }
    if (camerasData) {
      setCameras(camerasData)
    }
    setLoading(false)
  }

  // Load data only once on mount
  useEffect(() => {
    loadData()
    
    // Listen for refresh event from sync button
    const handleRefresh = () => {
      loadData(true)  // Force refresh when sync button is clicked
    }
    window.addEventListener('refresh-data', handleRefresh)
    
    return () => {
      window.removeEventListener('refresh-data', handleRefresh)
    }
  }, [])

  const { kpis = {}, queueStatus = {}, accuracyChart = [], yardUtilization = [] } = data
  
  // Use real camera data if available, otherwise use data from dashboard
  const cameraHealth = cameras.length > 0 ? cameras : data.cameraHealth
  
  // Update cameras online count with real data
  const camerasOnlineCount = cameras.length > 0 
    ? cameras.filter(c => c.status === 'online').length 
    : kpis.camerasOnline.value
  const camerasDegradedCount = cameras.length > 0
    ? cameras.filter(c => c.status === 'degraded').length
    : kpis.camerasOnline.degraded

  if (loading) {
    return (
      <div className="dashboard">
        <div className="loading-message">Loading dashboard data...</div>
      </div>
    )
  }

  return (
    <div className="dashboard">
      <div className="kpi-grid">
        <KPICard
          title="Trailers on Yard"
          value={kpis.trailersOnYard?.value || 0}
          subtitle={kpis.trailersOnYard?.change || '+0'}
          icon={kpis.trailersOnYard?.icon || '🚛'}
        />
        <KPICard
          title="New Detections (24h)"
          value={kpis.newDetections24h?.value || 0}
          subtitle={`OCR=${kpis.newDetections24h?.ocrAccuracy || '0%'}`}
          icon={kpis.newDetections24h?.icon || '📈'}
        />
        <KPICard
          title="Anomalies"
          value={kpis.anomalies?.value || 0}
          subtitle={kpis.anomalies?.description || 'No anomalies detected'}
          icon={kpis.anomalies?.icon || '⚠️'}
        />
        <KPICard
          title="Cameras Online"
          value={camerasOnlineCount}
          subtitle={`${camerasDegradedCount} degraded`}
          icon={kpis.camerasOnline?.icon || '📷'}
        />
      </div>

      <div className="dashboard-controls">
        <div className="search-bar">
          <Search className="search-icon" size={16} />
          <input type="text" placeholder="Search trailer, plate, spot..." />
        </div>
        <div className="filter-dropdown">
          <button className="filter-btn">
            All events <ChevronDown size={14} />
          </button>
        </div>
        <QueueStatus queueStatus={queueStatus} />
      </div>

      <div className="charts-grid">
        <div className="chart-card">
          <h3>Detection / OCR Accuracy (Today)</h3>
          <AccuracyChart data={accuracyChart} />
        </div>
        <div className="chart-card">
          <h3>Yard Utilization by Lane</h3>
          <YardUtilizationChart data={yardUtilization} />
        </div>
      </div>

      <RecentTrailerEvents onEventSelect={setSelectedEvent} />

      <YardMapSpotResolver selectedEvent={selectedEvent} />

      <div className="camera-health-section">
        <h3>Camera Health</h3>
        {cameras.length === 0 ? (
          <div className="no-cameras-message">
            No cameras connected. Make sure your video processing pipeline is running.
          </div>
        ) : (
          <CameraHealthTable cameras={cameraHealth} />
        )}
      </div>
    </div>
  )
}

export default Dashboard

