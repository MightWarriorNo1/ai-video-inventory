import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import './YardUtilizationChart.css'

const YardUtilizationChart = ({ data }) => {
  return (
    <div className="yard-utilization-chart">
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="lane" stroke="#6b7280" style={{ fontSize: '12px' }} />
          <YAxis 
            domain={[0, 100]} 
            stroke="#6b7280" 
            style={{ fontSize: '12px' }}
            label={{ value: 'Utilization %', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: 'white', border: '1px solid #e5e7eb', borderRadius: '6px' }}
            formatter={(value) => `${value}%`}
          />
          <Bar dataKey="utilization" fill="#1a1f3a" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default YardUtilizationChart










