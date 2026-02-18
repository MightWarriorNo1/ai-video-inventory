import { useState, useEffect } from 'react'
import { fetchVideoFrameRecords } from '../services/api'
import './VideoFrames.css'

const VideoFrames = () => {
  const [records, setRecords] = useState([])
  const [stats, setStats] = useState({ total: 0, unprocessed: 0, processed: 0 })
  const [loading, setLoading] = useState(true)
  const [filterStatus, setFilterStatus] = useState('all') // 'all', 'unprocessed', 'processed'
  const [filterCameraId, setFilterCameraId] = useState('')
  const [offset, setOffset] = useState(0)
  const [limit] = useState(50)

  const loadData = async () => {
    setLoading(true)
    try {
      const isProcessed = filterStatus === 'all' 
        ? null 
        : filterStatus === 'processed' ? true : false
      
      const data = await fetchVideoFrameRecords({
        limit,
        offset,
        is_processed: isProcessed,
        camera_id: filterCameraId || null
      })
      
      if (data) {
        setRecords(data.records || [])
        setStats(data.stats || { total: 0, unprocessed: 0, processed: 0 })
      }
    } catch (error) {
      console.error('Error loading video frame records:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    
    // Auto-refresh every 5 seconds to show new data from database
    const refreshInterval = setInterval(() => {
      loadData()
    }, 5000)
    
    // Listen for refresh event from sync button
    const handleRefresh = () => {
      loadData()
    }
    window.addEventListener('refresh-data', handleRefresh)
    
    return () => {
      clearInterval(refreshInterval)
      window.removeEventListener('refresh-data', handleRefresh)
    }
  }, [offset, filterStatus, filterCameraId])

  const formatGPS = (lat, lon) => {
    if (lat != null && lon != null) {
      return `${lat.toFixed(6)}, ${lon.toFixed(6)}`
    }
    return 'N/A'
  }

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A'
    try {
      return new Date(dateStr).toLocaleString()
    } catch (e) {
      return dateStr
    }
  }

  const handleFilterChange = (newStatus) => {
    setFilterStatus(newStatus)
    setOffset(0) // Reset to first page
  }

  const handleCameraFilterChange = (e) => {
    setFilterCameraId(e.target.value)
    setOffset(0) // Reset to first page
  }

  const handlePrevious = () => {
    if (offset > 0) {
      setOffset(Math.max(0, offset - limit))
    }
  }

  const handleNext = () => {
    if (records.length === limit) {
      setOffset(offset + limit)
    }
  }

  if (loading) {
    return (
      <div className="video-frames-page">
        <div className="loading-message">Loading video frame records...</div>
      </div>
    )
  }

  return (
    <div className="video-frames-page">
      <div className="video-frames-stats">
        <div className="stat-card">
          <div className="stat-label">Total Records</div>
          <div className="stat-value">{stats.total || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Unprocessed</div>
          <div className="stat-value">{stats.unprocessed || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Processed</div>
          <div className="stat-value">{stats.processed || 0}</div>
        </div>
      </div>

      <div className="video-frames-table-container">
        <div className="table-header">
          <h3>Video Frame Records</h3>
          <div className="table-actions">
            <select 
              value={filterStatus} 
              onChange={(e) => handleFilterChange(e.target.value)}
              className="filter-select"
            >
              <option value="all">All</option>
              <option value="unprocessed">Unprocessed</option>
              <option value="processed">Processed</option>
            </select>
            <input 
              type="text" 
              placeholder="Filter by Camera ID..." 
              value={filterCameraId}
              onChange={handleCameraFilterChange}
              className="search-input" 
            />
          </div>
        </div>
        <table className="video-frames-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>License Plate/Trailer</th>
              <th>GPS Coordinates</th>
              <th>Speed</th>
              <th>Barrier</th>
              <th>Confidence</th>
              <th>Camera ID</th>
              <th>Frame #</th>
              <th>Track ID</th>
              <th>Image Path</th>
              <th>Created On</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {records.length === 0 ? (
              <tr>
                <td colSpan="12" className="no-data">
                  No video frame records found.
                </td>
              </tr>
            ) : (
              records.map((record) => (
                <tr key={record.id}>
                  <td>{record.id}</td>
                  <td>{record.licence_plate_trailer || 'N/A'}</td>
                  <td className="gps-coords">{formatGPS(record.latitude, record.longitude)}</td>
                  <td>{record.speed != null ? record.speed.toFixed(2) : 'N/A'}</td>
                  <td>{record.barrier != null ? record.barrier.toFixed(2) : 'N/A'}</td>
                  <td>
                    <div className="confidence-bar">
                      <div 
                        className="confidence-fill" 
                        style={{ width: `${(record.confidence || 0) * 100}%` }}
                      />
                      <span>{((record.confidence || 0) * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td>{record.camera_id || 'N/A'}</td>
                  <td>{record.frame_number || 'N/A'}</td>
                  <td>{record.track_id != null ? record.track_id : 'N/A'}</td>
                  <td className="image-path">{record.image_path || 'N/A'}</td>
                  <td>{formatDate(record.created_on)}</td>
                  <td>
                    <span className={`status-badge ${record.is_processed ? 'processed' : 'unprocessed'}`}>
                      {record.is_processed ? 'Processed' : 'Unprocessed'}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
        <div className="pagination">
          <button 
            onClick={handlePrevious} 
            disabled={offset === 0}
            className="btn-pagination"
          >
            Previous
          </button>
          <span className="pagination-info">
            Showing {offset + 1} - {offset + records.length} of {stats.total || 0}
          </span>
          <button 
            onClick={handleNext} 
            disabled={records.length < limit}
            className="btn-pagination"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  )
}

export default VideoFrames
