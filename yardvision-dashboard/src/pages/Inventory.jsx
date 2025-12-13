import { useState, useEffect } from 'react'
import { fetchInventoryData } from '../services/api'
import { inventoryData as fallbackData } from '../data/mockData'
import './Inventory.css'

const Inventory = () => {
  const [data, setData] = useState(fallbackData)
  const [loading, setLoading] = useState(true)

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
              <th>Detected At</th>
              <th>OCR Confidence</th>
            </tr>
          </thead>
          <tbody>
            {trailers.length === 0 ? (
              <tr>
                <td colSpan="6" className="no-data">
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
                    <td>{detectedAtStr}</td>
                    <td>
                      <div className="confidence-bar">
                        <div 
                          className="confidence-fill" 
                          style={{ width: `${(trailer.ocrConfidence || 0) * 100}%` }}
                        />
                        <span>{((trailer.ocrConfidence || 0) * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default Inventory

