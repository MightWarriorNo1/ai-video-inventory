import { configurationData } from '../data/mockData'
import { useState } from 'react'
import './Configuration.css'

const Configuration = () => {
  const [config, setConfig] = useState(configurationData)

  const handleCameraToggle = (cameraId) => {
    setConfig({
      ...config,
      cameras: config.cameras.map(cam =>
        cam.id === cameraId ? { ...cam, enabled: !cam.enabled } : cam
      )
    })
  }

  const handleProcessingChange = (key, value) => {
    setConfig({
      ...config,
      processing: { ...config.processing, [key]: value }
    })
  }

  return (
    <div className="configuration-page">
      <div className="config-section">
        <h3>Camera Configuration</h3>
        <div className="cameras-list">
          {config.cameras.map((camera) => (
            <div key={camera.id} className="camera-config-item">
              <div className="camera-info">
                <div className="camera-name-id">
                  <strong>{camera.name}</strong>
                  <span className="camera-id">{camera.id}</span>
                </div>
                <div className="camera-url">{camera.rtspUrl}</div>
              </div>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={camera.enabled}
                  onChange={() => handleCameraToggle(camera.id)}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
          ))}
        </div>
      </div>

      <div className="config-section">
        <h3>Processing Configuration</h3>
        <div className="processing-config">
          <div className="config-item">
            <label>Detect Every N Frames</label>
            <input
              type="number"
              value={config.processing.detectEveryN}
              onChange={(e) => handleProcessingChange('detectEveryN', parseInt(e.target.value))}
            />
          </div>
          <div className="config-item">
            <label>OCR Confidence Threshold</label>
            <input
              type="number"
              step="0.1"
              value={config.processing.ocrConfidenceThreshold}
              onChange={(e) => handleProcessingChange('ocrConfidenceThreshold', parseFloat(e.target.value))}
            />
          </div>
          <div className="config-item">
            <label>Tracker Threshold</label>
            <input
              type="number"
              step="0.1"
              value={config.processing.trackerThreshold}
              onChange={(e) => handleProcessingChange('trackerThreshold', parseFloat(e.target.value))}
            />
          </div>
        </div>
      </div>

      <div className="config-section">
        <h3>Integration Configuration</h3>
        <div className="integrations-config">
          <div className="integration-item">
            <div className="integration-info">
              <strong>MQTT</strong>
              <span>{config.integrations.mqtt.broker}</span>
            </div>
            <label className="toggle-switch">
              <input type="checkbox" checked={config.integrations.mqtt.enabled} readOnly />
              <span className="toggle-slider"></span>
            </label>
          </div>
          <div className="integration-item">
            <div className="integration-info">
              <strong>Kafka</strong>
              <span>{config.integrations.kafka.broker}</span>
            </div>
            <label className="toggle-switch">
              <input type="checkbox" checked={config.integrations.kafka.enabled} readOnly />
              <span className="toggle-slider"></span>
            </label>
          </div>
          <div className="integration-item">
            <div className="integration-info">
              <strong>Azure Service Bus</strong>
              <span>{config.integrations.azure.connectionString || 'Not configured'}</span>
            </div>
            <label className="toggle-switch">
              <input type="checkbox" checked={config.integrations.azure.enabled} readOnly />
              <span className="toggle-slider"></span>
            </label>
          </div>
        </div>
      </div>

      <div className="config-actions">
        <button className="btn-save">Save Configuration</button>
        <button className="btn-reset">Reset to Defaults</button>
      </div>
    </div>
  )
}

export default Configuration



