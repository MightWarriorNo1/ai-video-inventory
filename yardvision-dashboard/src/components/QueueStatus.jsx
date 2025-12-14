import './QueueStatus.css'

const QueueStatus = ({ queueStatus }) => {
  return (
    <div className="queue-status">
      <div className="queue-item">
        <span className="queue-label">IngestQ:</span>
        <span className="queue-value">{queueStatus.ingestQ}</span>
      </div>
      <div className="queue-item">
        <span className="queue-label">OCRQ:</span>
        <span className="queue-value">{queueStatus.ocrQ}</span>
      </div>
      <div className="queue-item">
        <span className="queue-label">PubQ:</span>
        <span className="queue-value">{queueStatus.pubQ}</span>
      </div>
    </div>
  )
}

export default QueueStatus



