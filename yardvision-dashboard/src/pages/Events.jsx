import { useState, useEffect } from 'react'
import { fetchDashboardEvents } from '../services/api'
import './Events.css'

const Events = () => {
  const [selectedFilter, setSelectedFilter] = useState('All events')
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)
  
  const filters = ['All events', 'Detection', 'OCR', 'Anomaly', 'Spot Changes']

  useEffect(() => {
    const loadEvents = async () => {
      setLoading(true)
      const eventsData = await fetchDashboardEvents(1000)
      // Transform events to match expected format
      const transformedEvents = eventsData.map((event, index) => ({
        id: index + 1,
        timestamp: event.ts_iso || '',
        camera: event.camera_id || 'N/A',
        trackId: event.track_id || 'N/A',
        plate: event.text || 'N/A',
        spot: event.spot || 'unknown',
        type: event.text ? 'OCR' : (event.conf && parseFloat(event.conf) < 0.5 ? 'Anomaly' : 'Detection'),
        confidence: parseFloat(event.conf || 0)
      }))
      setEvents(transformedEvents)
      setLoading(false)
    }
    
    loadEvents()
    // Refresh every 5 seconds
    const interval = setInterval(loadEvents, 5000)
    return () => clearInterval(interval)
  }, [])

  const filteredEvents = selectedFilter === 'All events' 
    ? events 
    : events.filter(e => e.type === selectedFilter)

  if (loading) {
    return (
      <div className="events-page">
        <div className="loading-message">Loading events...</div>
      </div>
    )
  }

  return (
    <div className="events-page">
      <div className="events-header">
        <h2>Events Log</h2>
        <div className="events-controls">
          <select 
            value={selectedFilter} 
            onChange={(e) => setSelectedFilter(e.target.value)}
            className="filter-select"
          >
            {filters.map(filter => (
              <option key={filter} value={filter}>{filter}</option>
            ))}
          </select>
          <button className="btn-refresh">Refresh</button>
          <button className="btn-export">Export</button>
        </div>
      </div>

      <div className="events-table-container">
        <table className="events-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Camera</th>
              <th>Track ID</th>
              <th>Plate</th>
              <th>Spot</th>
              <th>Type</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {filteredEvents.map((event) => (
              <tr key={event.id}>
                <td>{new Date(event.timestamp).toLocaleString()}</td>
                <td>{event.camera}</td>
                <td>{event.trackId}</td>
                <td>{event.plate}</td>
                <td>{event.spot}</td>
                <td>
                  <span className={`event-type-badge ${event.type.toLowerCase()}`}>
                    {event.type}
                  </span>
                </td>
                <td>
                  <div className="confidence-value">
                    {(event.confidence * 100).toFixed(1)}%
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

export default Events



