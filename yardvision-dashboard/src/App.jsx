import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Inventory from './pages/Inventory'
import YardView from './pages/YardView'
import CameraHealth from './pages/CameraHealth'
import Events from './pages/Events'
import Configuration from './pages/Configuration'
import Reports from './pages/Reports'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/inventory" element={<Inventory />} />
          <Route path="/yard-view" element={<YardView />} />
          <Route path="/camera-health" element={<CameraHealth />} />
          <Route path="/events" element={<Events />} />
          <Route path="/configuration" element={<Configuration />} />
          <Route path="/reports" element={<Reports />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App











