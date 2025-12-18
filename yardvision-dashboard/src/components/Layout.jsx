import { Link, useLocation } from 'react-router-dom'
import { 
  LayoutDashboard, 
  Package, 
  MapPin, 
  Settings, 
  FileText, 
  SlidersHorizontal, 
  BarChart3,
  RefreshCw,
  Download
} from 'lucide-react'
import './Layout.css'

const Layout = ({ children }) => {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/inventory', label: 'Inventory', icon: Package },
    { path: '/yard-view', label: 'Yard View', icon: MapPin },
    { path: '/camera-health', label: 'Camera Health', icon: Settings },
    { path: '/events', label: 'Events', icon: FileText },
    { path: '/configuration', label: 'Configuration', icon: SlidersHorizontal },
    { path: '/reports', label: 'Reports', icon: BarChart3 },
  ]

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>YardVision AI</h1>
        </div>
        <nav className="sidebar-nav">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path || 
              (item.path === '/' && location.pathname === '/') ||
              (item.path !== '/' && location.pathname.startsWith(item.path))
            
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`nav-item ${isActive ? 'active' : ''}`}
              >
                <Icon className="nav-icon" size={18} />
                <span className="nav-label">{item.label}</span>
              </Link>
            )
          })}
        </nav>
      </aside>
      <main className="main-content">
        <header className="top-header">
          <div className="header-left">
            <div className="logo-circle">P</div>
            <div>
              <h2>Prosper YardVision AI — Video Inventory</h2>
              <p className="subtitle">
                Computer vision for the yard • YOLO (TensorRT) • ByteTrack • OCR • Homography Spot Resolver
              </p>
            </div>
          </div>
          <div className="header-actions">
            <button 
              className="btn-sync"
              onClick={() => {
                // Trigger refresh event for all pages
                window.dispatchEvent(new Event('refresh-data'))
              }}
            >
              <RefreshCw size={16} />
              Sync
            </button>
            <button className="btn-export">
              <Download size={16} />
              Export CSV
            </button>
          </div>
        </header>
        <div className="content-area">
          {children}
        </div>
      </main>
    </div>
  )
}

export default Layout










