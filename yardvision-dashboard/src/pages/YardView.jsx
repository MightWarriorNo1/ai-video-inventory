import { useState, useEffect } from 'react'
import { fetchYardViewData } from '../services/api'
import { yardViewData as fallbackData } from '../data/mockData'
import './YardView.css'

const YardView = () => {
  const [data, setData] = useState(fallbackData)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      const res = await fetchYardViewData()
      if (res && (res.spots?.length > 0 || res.lanes?.length > 0)) {
        setData(res)
      }
      setLoading(false)
    }
    load()
    const handleRefresh = () => load()
    window.addEventListener('refresh-data', handleRefresh)
    return () => window.removeEventListener('refresh-data', handleRefresh)
  }, [])

  const { spots, lanes } = data

  const getSpotsByLane = (lane) => {
    return spots.filter(spot => spot.lane === lane)
  }

  if (loading) {
    return (
      <div className="yard-view-page">
        <div className="loading-message">Loading yard view...</div>
      </div>
    )
  }

  return (
    <div className="yard-view-page">
      <div className="yard-view-header">
        <h2>Yard Layout</h2>
        <div className="legend">
          <div className="legend-item">
            <div className="legend-color occupied"></div>
            <span>Occupied</span>
          </div>
          <div className="legend-item">
            <div className="legend-color available"></div>
            <span>Available</span>
          </div>
        </div>
      </div>

      <div className="yard-layout">
        {lanes.map((lane) => {
          const laneSpots = getSpotsByLane(lane)
          return (
            <div key={lane} className="lane-section">
              <h3 className="lane-title">Lane {lane}</h3>
              <div className="spots-grid">
                {laneSpots.map((spot) => (
                  <div
                    key={spot.id}
                    className={`spot-card ${spot.occupied ? 'occupied' : 'available'}`}
                  >
                    <div className="spot-id">{spot.id}</div>
                    {spot.occupied ? (
                      <>
                        <div className="spot-trailer-id">{spot.trailerId}</div>
                        <div className="spot-plate">{spot.plate}</div>
                      </>
                    ) : (
                      <div className="spot-empty">Empty</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default YardView














