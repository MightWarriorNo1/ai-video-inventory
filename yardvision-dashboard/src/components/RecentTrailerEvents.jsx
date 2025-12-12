import { useState, useEffect } from 'react'
import { fetchDashboardEvents } from '../services/api'
import './RecentTrailerEvents.css'

const RecentTrailerEvents = () => {
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedEvent, setSelectedEvent] = useState(null)

  useEffect(() => {
    const loadEvents = async () => {
      setLoading(true)
      const eventsData = await fetchDashboardEvents(50) // Get last 50 events
      
      // Transform and sort by timestamp (most recent first)
      const transformedEvents = eventsData
        .map((event, index) => ({
          id: index + 1,
          timestamp: event.ts_iso || '',
          trailer: event.text || 'N/A',
          plate: event.text || 'N/A',
          spot: event.spot || 'unknown',
          conf: parseFloat(event.conf || 0) * 100,
          ocr: parseFloat(event.conf || 0) * 100,
          source: event.camera_id || 'N/A',
          rawEvent: event
        }))
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
        .slice(0, 10) // Show only last 10
      
      setEvents(transformedEvents)
      if (transformedEvents.length > 0 && !selectedEvent) {
        setSelectedEvent(transformedEvents[0])
      }
      setLoading(false)
    }
    
    loadEvents()
    const interval = setInterval(loadEvents, 5000)
    return () => clearInterval(interval)
  }, [])

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

  if (loading) {
    return (
      <div className="recent-trailer-events">
        <h3>Recent Trailer Events</h3>
        <div className="loading-message">Loading events...</div>
      </div>
    )
  }

  return (
    <div className="recent-trailer-events">
      <h3>Recent Trailer Events</h3>
      <div className="events-table-wrapper">
        <table className="events-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Trailer</th>
              <th>Plate</th>
              <th>Spot</th>
              <th>Conf</th>
              <th>OCR</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {events.length === 0 ? (
              <tr>
                <td colSpan="7" className="no-events">No events found</td>
              </tr>
            ) : (
              events.map((event) => (
                <tr 
                  key={event.id} 
                  className={selectedEvent?.id === event.id ? 'selected' : ''}
                  onClick={() => {
                    setSelectedEvent(event)
                    if (onEventSelect) {
                      onEventSelect(event)
                    }
                  }}
                >
                  <td>{formatTime(event.timestamp)}</td>
                  <td className="trailer-link">{event.trailer}</td>
                  <td>{event.plate}</td>
                  <td>
                    <span className="spot-badge">{event.spot}</span>
                  </td>
                  <td>{event.conf.toFixed(0)}%</td>
                  <td>{event.ocr.toFixed(0)}%</td>
                  <td>{event.source}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default RecentTrailerEvents
