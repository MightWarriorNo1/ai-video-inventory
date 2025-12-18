import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import './AccuracyChart.css'

const AccuracyChart = ({ data }) => {
  return (
    <div className="accuracy-chart">
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="time" stroke="#6b7280" style={{ fontSize: '12px' }} />
          <YAxis 
            domain={[80, 100]} 
            stroke="#6b7280" 
            style={{ fontSize: '12px' }}
            label={{ value: 'Accuracy %', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: 'white', border: '1px solid #e5e7eb', borderRadius: '6px' }}
            formatter={(value) => `${value}%`}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="detection" 
            stroke="#3b82f6" 
            strokeWidth={2}
            name="Detection"
            dot={{ r: 3 }}
          />
          <Line 
            type="monotone" 
            dataKey="ocr" 
            stroke="#10b981" 
            strokeWidth={2}
            name="OCR"
            dot={{ r: 3 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default AccuracyChart














