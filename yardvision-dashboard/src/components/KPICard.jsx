import { Truck, TrendingUp, AlertTriangle, Camera } from 'lucide-react'
import './KPICard.css'

const iconMap = {
  '🚛': Truck,
  '📈': TrendingUp,
  '⚠️': AlertTriangle,
  '📷': Camera,
}

const KPICard = ({ title, value, subtitle, icon }) => {
  const IconComponent = iconMap[icon] || Truck
  
  return (
    <div className="kpi-card">
      <div className="kpi-icon">
        <IconComponent size={32} />
      </div>
      <div className="kpi-content">
        <div className="kpi-title">{title}</div>
        <div className="kpi-value">{value}</div>
        <div className="kpi-subtitle">{subtitle}</div>
      </div>
    </div>
  )
}

export default KPICard










