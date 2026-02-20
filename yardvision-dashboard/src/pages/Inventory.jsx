import { useState, useEffect } from 'react'
import { fetchInventoryData } from '../services/api'
import { inventoryData as fallbackData } from '../data/mockData'
import './Inventory.css'

const Inventory = () => {
  const [data, setData] = useState(fallbackData)
  const [loading, setLoading] = useState(true)
  const [imageModalUrl, setImageModalUrl] = useState(null)

  const loadData = async () => {
    setLoading(true)
    const inventoryData = await fetchInventoryData()
    if (inventoryData) {
      setData(inventoryData)
    }
    setLoading(false)
  }

  useEffect(() => {
    loadData()
    
    // Listen for refresh event from sync button
    const handleRefresh = () => {
      loadData()
    }
    window.addEventListener('refresh-data', handleRefresh)
    
    return () => {
      window.removeEventListener('refresh-data', handleRefresh)
    }
  }, [])

  if (loading) {
    return (
      <div className="inventory-page">
        <div className="loading-message">Loading inventory data...</div>
      </div>
    )
  }

  const { trailers, stats } = data

  // Normalize confidence to 0–100 (API may send 0–1 or 0–100)
  const confidencePct = (val) => {
    if (val == null) return 0
    const pct = val > 1 ? val : val * 100
    return Math.min(100, Math.max(0, pct))
  }

  return (
    <div className="inventory-page">
      <div className="inventory-stats">
        <div className="stat-card">
          <div className="stat-label">Total Trailers</div>
          <div className="stat-value">{stats.total}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Parked</div>
          <div className="stat-value">{stats.parked}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">In Transit</div>
          <div className="stat-value">{stats.inTransit}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Anomalies</div>
          <div className="stat-value">{stats.anomalies}</div>
        </div>
      </div>

      <div className="inventory-table-container">
        <div className="table-header">
          <h3>Trailer Inventory</h3>
          <div className="table-actions">
            <input type="text" placeholder="Search..." className="search-input" />
            <button className="btn-filter">Filter</button>
            <button className="btn-export">Export</button>
          </div>
        </div>
        <table className="inventory-table">
          <thead>
            <tr>
              <th>Trailer ID</th>
              <th>Plate Number</th>
              <th>Spot</th>
              <th>Status</th>
              <th>GPS Coordinates</th>
              <th>Detected At</th>
              <th>OCR Confidence</th>
              <th>Image</th>
            </tr>
          </thead>
          <tbody>
            {trailers.length === 0 ? (
              <tr>
                <td colSpan="8" className="no-data">
                  No trailers found. Make sure combined_results.json files exist in out/crops/test-video/ folders.
                </td>
              </tr>
            ) : (
              trailers.map((trailer) => {
                let detectedAtStr = 'N/A'
                try {
                  if (trailer.detectedAt) {
                    detectedAtStr = new Date(trailer.detectedAt).toLocaleString()
                  }
                } catch (e) {
                  detectedAtStr = trailer.detectedAt || 'N/A'
                }
                
                const formatGPS = (lat, lon) => {
                  if (lat != null && lon != null) {
                    return `${lat.toFixed(6)}, ${lon.toFixed(6)}`
                  }
                  return 'N/A'
                }
                
                return (
                  <tr key={trailer.id}>
                    <td>{trailer.id}</td>
                    <td>{trailer.plate || 'N/A'}</td>
                    <td>{trailer.spot || 'N/A'}</td>
                    <td>
                      <span className={`status-badge ${(trailer.status || 'unknown').toLowerCase().replace(' ', '-')}`}>
                        {trailer.status || 'Unknown'}
                      </span>
                    </td>
                    <td className="gps-coords">{formatGPS(trailer.lat, trailer.lon)}</td>
                    <td>{detectedAtStr}</td>
                    <td>
                      <div className="confidence-bar">
                        <div className="confidence-track">
                          <div
                            className="confidence-fill"
                            style={{ width: `${confidencePct(trailer.ocrConfidence)}%` }}
                          />
                        </div>
                        <span>{confidencePct(trailer.ocrConfidence).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td>
                      {trailer.imageUrl ? (
                        <button
                          type="button"
                          className="btn-view-image"
                          onClick={() => setImageModalUrl(trailer.imageUrl)}
                        >
                          View
                        </button>
                      ) : (
                        <span className="no-image">—</span>
                      )}
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>

      {imageModalUrl && (
        <div className="image-modal-overlay" onClick={() => setImageModalUrl(null)} role="presentation">
          <div className="image-modal" onClick={(e) => e.stopPropagation()}>
            <button type="button" className="image-modal-close" onClick={() => setImageModalUrl(null)} aria-label="Close">
              ×
            </button>
            <img src={imageModalUrl} alt="Cropped trailer" className="image-modal-img" />
          </div>
        </div>
      )}
    </div>
  )
}

export default Inventory








