import { useState, useEffect } from 'react'
import { fetchInventoryData } from '../services/api'
import { inventoryData as fallbackData } from '../data/mockData'
import './Inventory.css'

const Inventory = () => {
  const [data, setData] = useState(fallbackData)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      const inventoryData = await fetchInventoryData()
      if (inventoryData) {
        setData(inventoryData)
      }
      setLoading(false)
    }
    
    loadData()
    // Refresh every 10 seconds
    const interval = setInterval(loadData, 10000)
    return () => clearInterval(interval)
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
            {trailers.map((trailer) => (
              <tr key={trailer.id}>
                <td>{trailer.id}</td>
                <td>{trailer.plate}</td>
                <td>{trailer.spot}</td>
                <td>
                  <span className={`status-badge ${trailer.status.toLowerCase()}`}>
                    {trailer.status}
                  </span>
                </td>
                <td>{new Date(trailer.detectedAt).toLocaleString()}</td>
                <td>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill" 
                      style={{ width: `${trailer.ocrConfidence * 100}%` }}
                    />
                    <span>{(trailer.ocrConfidence * 100).toFixed(1)}%</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default Inventory

