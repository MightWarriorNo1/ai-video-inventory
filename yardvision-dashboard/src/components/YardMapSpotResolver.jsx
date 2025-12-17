import { useState, useEffect } from 'react'
import { fetchDashboardEvents } from '../services/api'
import './YardMapSpotResolver.css'

const YardMapSpotResolver = ({ selectedEvent }) => {
  const [homographyQuality, setHomographyQuality] = useState(0.97)

  useEffect(() => {
    // Calculate homography quality from selected event
    if (selectedEvent?.rawEvent) {
      const conf = parseFloat(selectedEvent.rawEvent.conf || 0)
      const method = selectedEvent.rawEvent.method || ''
      
      // Higher quality if confidence is high and method is good
      let quality = conf
      if (method === 'polygon' || method === 'calibration') {
        quality = Math.min(0.99, conf + 0.1)
      }
      setHomographyQuality(quality)
    }
  }, [selectedEvent])

  const formatTime = (timestamp) => {
    if (!timestamp) return 'N/A'
    try {
      const date = new Date(timestamp)
      return date.toLocaleTimeString('en-US', { 
        hour12: false, 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
      })
    } catch {
      return timestamp
    }
  }

  return (
    <div className="yard-map-spot-resolver">
      <h3>Yard Map / Spot Resolver</h3>
      <div className="yard-map-container">
        <div className="map-placeholder">
          <p>(Embed your yard SVG / Leaflet map here with spots A1...Z99)</p>
        </div>
        <div className="selected-trailer-details">
          <h4>Selected Trailer</h4>
          {selectedEvent ? (
            <div className="trailer-info">
              <div className="info-row">
                <span className="info-label">Trailer:</span>
                <span className="info-value trailer-id">{selectedEvent.trailer}</span>
              </div>
              <div className="info-row">
                <span className="info-label">Plate:</span>
                <span className="info-value">{selectedEvent.plate}</span>
              </div>
              <div className="info-row">
                <span className="info-label">Resolved Spot:</span>
                <span className="spot-badge-large">{selectedEvent.spot}</span>
              </div>
              <div className="info-row">
                <span className="info-label">Homography Quality:</span>
                <span className="quality-badge">{homographyQuality.toFixed(2)}</span>
              </div>
              <div className="info-row">
                <span className="info-label">Last Seen:</span>
                <span className="info-value">
                  {formatTime(selectedEvent.timestamp)} ({selectedEvent.source})
                </span>
              </div>
            </div>
          ) : (
            <div className="no-selection">
              <p>Select a trailer from the events table to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default YardMapSpotResolver









