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
            <th>Type</th>
            <th>Source</th>
            <th>Status</th>
            <th>FPS</th>
            <th>Latency</th>
          </tr>
        </thead>
        <tbody>
          {cameras.length === 0 ? (
            <tr>
              <td colSpan="7" className="no-cameras">
                No cameras configured. Check cameras.yaml configuration file.
              </td>
            </tr>
          ) : (
            cameras.map((camera) => (
              <tr key={camera.id}>
                <td>
                  <Camera className="camera-icon" size={16} />
                  {camera.name || camera.id}
                </td>
                <td>{camera.id}</td>
                <td>
                  <span className={`type-badge ${camera.type?.toLowerCase() || 'unknown'}`}>
                    {camera.type || 'Unknown'}
                  </span>
                </td>
                <td className="source-cell">
                  {camera.rtsp_url || 'N/A'}
                </td>
                <td>
                  <span className={`status-badge ${camera.status || 'offline'}`}>
                    {camera.status || 'offline'}
                  </span>
                </td>
                <td>{camera.fps || 0}</td>
                <td>{camera.latency || 0}ms</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}

export default CameraHealthTable










