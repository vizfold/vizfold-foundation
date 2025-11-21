import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import HomePage from './pages/HomePage'
import ProteinsPage from './pages/ProteinsPage'
import VisualizationPage from './pages/VisualizationPage'
import InferencePage from './pages/InferencePage'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/proteins" element={<ProteinsPage />} />
          <Route path="/proteins/:proteinId/visualize" element={<VisualizationPage />} />
          <Route path="/inference" element={<InferencePage />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App
