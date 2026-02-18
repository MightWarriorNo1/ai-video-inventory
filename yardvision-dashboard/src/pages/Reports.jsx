import { useState, useEffect } from 'react'
import { fetchReportsData } from '../services/api'
import { reportsData as fallbackData } from '../data/mockData'
import './Reports.css'

const Reports = () => {
  const [data, setData] = useState(fallbackData)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      const res = await fetchReportsData()
      if (res) {
        setData(res)
      }
      setLoading(false)
    }
    load()
    const handleRefresh = () => load()
    window.addEventListener('refresh-data', handleRefresh)
    return () => window.removeEventListener('refresh-data', handleRefresh)
  }, [])

  const { daily, weekly, monthly } = data

  const ReportCard = ({ title, data }) => (
    <div className="report-card">
      <h3>{title}</h3>
      <div className="report-stats">
        <div className="report-stat">
          <div className="report-stat-label">Total Detections</div>
          <div className="report-stat-value">{(data.totalDetections ?? 0).toLocaleString()}</div>
        </div>
        <div className="report-stat">
          <div className="report-stat-label">OCR Accuracy</div>
          <div className="report-stat-value">{data.ocrAccuracy ?? 0}%</div>
        </div>
        <div className="report-stat">
          <div className="report-stat-label">Anomalies</div>
          <div className="report-stat-value">{data.anomalies ?? 0}</div>
        </div>
        <div className="report-stat">
          <div className="report-stat-label">Avg Processing Time</div>
          <div className="report-stat-value">{data.avgProcessingTime ?? 0}ms</div>
        </div>
      </div>
    </div>
  )

  if (loading) {
    return (
      <div className="reports-page">
        <div className="loading-message">Loading reports...</div>
      </div>
    )
  }

  return (
    <div className="reports-page">
      <div className="reports-header">
        <h2>Reports</h2>
        <div className="reports-actions">
          <button className="btn-generate">Generate Report</button>
          <button className="btn-export">Export All</button>
        </div>
      </div>

      <div className="reports-grid">
        <ReportCard title={`Daily Report - ${daily.date}`} data={daily} />
        <ReportCard title={`Weekly Report - ${weekly.week}`} data={weekly} />
        <ReportCard title={`Monthly Report - ${monthly.month}`} data={monthly} />
      </div>

      <div className="reports-summary">
        <h3>Summary</h3>
        <div className="summary-content">
          <p>
            The system has processed <strong>{(monthly.totalDetections ?? 0).toLocaleString()}</strong> detections 
            this month with an average OCR accuracy of <strong>{monthly.ocrAccuracy ?? 0}%</strong>.
          </p>
          <p>
            Average processing time is <strong>{monthly.avgProcessingTime ?? 0}ms</strong> per frame, 
            with <strong>{monthly.anomalies ?? 0}</strong> anomalies detected requiring attention.
          </p>
        </div>
      </div>
    </div>
  )
}

export default Reports














