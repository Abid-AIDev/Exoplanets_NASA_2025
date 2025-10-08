import React from 'react'
import Header from './components/Header'
import PlanetInfo from './components/PlanetInfo'
import './App.css'
import { usePlanet } from './context/PlanetContext.jsx'

const backgroundColors = [
  '#1a237e', // Indigo
  '#283593', // Deep Indigo
  '#1565c0', // Blue
  '#00838f', // Teal
  '#00695c', // Green
  '#4527a0', // Purple
  '#ad1457', // Pink
  '#c62828', // Red
  '#f9a825', // Yellow
  '#ef6c00', // Orange
  '#37474f', // Blue Grey
  '#263238', // Dark Blue Grey
  '#212121', // Almost Black
  '#607d8b', // Muted Blue Grey
  '#3949ab', // Blue Accent
];

function App() {
  const { bgColor } = usePlanet()
  return (
    <div className="app" style={{ background: bgColor, minHeight: '100vh' }}>
      <Header />
      <PlanetInfo />
      <div className="planet-container">
        <div className="planet-image" aria-hidden="true"></div>
      </div>
    </div>
  )
}

export default App
