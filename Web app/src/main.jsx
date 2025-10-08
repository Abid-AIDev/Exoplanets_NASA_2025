import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'

import App from './App.jsx';
import Characteristics from './components/Characteristics.jsx';
import ErrorBoundary from './components/ErrorBoundary.jsx';
import { PlanetProvider } from './context/PlanetContext.jsx';
import Exoplanet3DPage from './components/pages/Exoplanet3DPage.jsx';

import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary>
      <PlanetProvider>
        <Router>
          <Routes>
            <Route path="/" element={<App />} />
            <Route path="/characteristics" element={<Characteristics />} />
            <Route path="/exoplanet-3d" element={<Exoplanet3DPage />} />
            
          </Routes>
        </Router>
      </PlanetProvider>
    </ErrorBoundary>
  </React.StrictMode>,
)

