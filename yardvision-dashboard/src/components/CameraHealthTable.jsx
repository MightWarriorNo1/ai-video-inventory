import { Camera } from 'lucide-react'
import './CameraHealthTable.css'

const CameraHealthTable = ({ cameras }) => {
  return (
    <div className="camera-health-table">
      <table>
        <thead>
          <tr>
            <th>Camera</th>
            <th>Camera ID</th>
            <th>FPS</th>
            <th>Latency</th>
            <th>Uptime</th>
          </tr>
        </thead>
        <tbody>
          {cameras.map((camera) => (
            <tr key={camera.id}>
              <td>
                <Camera className="camera-icon" size={16} />
                {camera.name}
              </td>
              <td>{camera.id}</td>
              <td>{camera.fps}</td>
              <td>{camera.latency}ms</td>
              <td>
                <div className="uptime-container">
                  <div className="uptime-bar">
                    <div 
                      className="uptime-fill" 
                      style={{ width: `${camera.uptime}%` }}
                    />
                  </div>
                  <span className={`status-badge ${camera.status}`}>
                    {camera.status}
                  </span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default CameraHealthTable
