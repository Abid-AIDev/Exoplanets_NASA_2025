import React, { useCallback, memo, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import '../Characteristics.css'
import { usePlanet } from '../context/PlanetContext'

// Loading component
const LoadingSpinner = memo(() => (
  <div className="loading-container">
    <div className="loading-spinner"></div>
    <p>Loading planet data...</p>
  </div>
))

// Error component
const ErrorMessage = memo(({ error, onRetry }) => (
  <div className="error-container">
    <div className="error-icon">⚠️</div>
    <h3>Oops! Something went wrong</h3>
    <p>{error}</p>
    <button className="retry-btn" onClick={onRetry}>
      Try Again
    </button>
  </div>
))

// Characteristic item component
const CharacteristicItem = memo(({ name, value }) => (
  <div className="characteristic-group">
    <div className="characteristic-name">{name}</div>
    <div className="characteristic-detail">{value || 'N/A'}</div>
  </div>
))

// Back button component
const BackButton = memo(({ onGoBack }) => (
  <button className="characteristics-btn" onClick={onGoBack} aria-label="Go back to overview">
    <span>←</span> Back to Overview
  </button>
))

const Characteristics = memo(() => {
  const navigate = useNavigate()
  const { currentPlanet, loading, error, planets } = usePlanet()
  
  const goBack = useCallback(() => {
    navigate('/')
  }, [navigate])

  const handleRetry = useCallback(() => {
    window.location.reload()
  }, [])

  // Memoized characteristic data
        const characteristics = useMemo(() => [
          ['depth_ppm', 'Depth (ppm)'],
          ['duration_hours', 'Duration (hours)'],
          ['duty_cycle', 'Duty Cycle'],
          ['label', 'Label'],
          ['label_raw', 'Label Raw'],
          ['mission', 'Mission'],
          ['mission_encoded', 'Mission Encoded'],
          ['object_id', 'Object ID'],
          ['period_days', 'Period (days)'],
          ['planet_radius_from_depth', 'Planet Radius (from depth)'],
          ['planet_radius_rearth', 'Planet Radius (Earth radii)'],
          ['score', 'Score'],
          ['transit_frequency', 'Transit Frequency']
        ].map(([key, label]) => ({
          name: label,
          value: currentPlanet?.[key]
        })), [currentPlanet])


  // State for images

  const [images] = useState([
    'fig1_linear_regression.png',
    'fig2_logistic_regression.png',
    'fig3_pca_kmeans.png',
    'fig4_decision_tree.png',
    'fig4_decision_tree_importance.png',
    'fig5_random_forest_importance.png',
    'fig6_correlation_heatmap.png',
    'fig7_violin_plots.png',
  ])
  const [imgError, setImgError] = useState(null)
  const [modalImg, setModalImg] = useState(null)

  // Optionally, you could fetch the list from the backend if you add a route for it

  if (loading) {
    return <LoadingSpinner />
  }

  if (error) {
    return <ErrorMessage error={error} onRetry={handleRetry} />
  }

  if (!currentPlanet) {
    return (
      <div className="no-data-container">
        <div className="no-data-icon">🌌</div>
        <h3>No planet data available</h3>
        <p>There is no planet data to display at the moment.</p>
        <BackButton onGoBack={goBack} />
      </div>
    )
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <a href="/" className="logo-link" aria-label="Home">Delta-V</a>
          </div>
          <nav className="header-nav" role="navigation" aria-label="Main navigation">
            <a href="#" className="nav-link">Planets</a>
            <a href="#" className="nav-link">Explore</a>
            <a href="#" className="nav-link">About us</a>
          </nav>
          <button
              className="donate-btn"
              aria-label="Redirects to donation page"
              onClick={() => window.location.href = "https://deddee362911.ngrok-free.app"}
            >
              Predict
            </button>
        </div>
      </header>

      <div className="characteristics-page">
        <div className="characteristics-content">
          <div className="description-column">
            <h1 className="characteristics-title">
              {currentPlanet.object_id || 'Unknown Planet'}
            </h1>
            <p className="characteristics-description">
              {currentPlanet.description || 'No description available for this planet.'}
            </p>
            <div className="characteristics-buttons">
              <BackButton onGoBack={goBack} />
            </div>

            <div className="output-images-section" style={{ marginTop: 32 }}>
              <h3 style={{ marginBottom: 20 }}>Exoplanet Analysis Visualizations</h3>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                  gap: 32,
                  justifyItems: 'center',
                }}
              >
                {images.map((img, idx) => (
                  <div
                    key={img}
                    style={{
                      background: 'rgba(255,255,255,0.15)',
                      borderRadius: 18,
                      boxShadow: '0 4px 24px 0 rgba(0,0,0,0.10)',
                      backdropFilter: 'blur(8px)',
                      border: '1.5px solid rgba(255,255,255,0.25)',
                      padding: 18,
                      width: '100%',
                      maxWidth: 340,
                      cursor: 'pointer',
                      transition: 'transform 0.15s',
                    }}
                    onClick={() => {
                      if (img === 'fig1_linear_regression.png') {
                        // Save only minimal planet data to localStorage for index.html
                        const minimalPlanets = planets.map(p => ({
                          period_days: p.period_days,
                          depth_ppm: p.depth_ppm,
                          planet_radius_rearth: p.planet_radius_rearth,
                          label_raw: p.label_raw
                        }));
                        localStorage.setItem('planets_data', JSON.stringify(minimalPlanets));
                        window.open('/server/index.html', '_blank');
                      } else {
                        setModalImg(img);
                      }
                    }}
                  >
                    <img
                      src={`http://localhost:5500/output-image?filename=${img}`}
                      onError={(e) => {
                        const s = e?.target?.src || ''
                        // Try falling back from ports that might be blocked (5000/5001) to 5500
                        if (s.includes(':5000') || s.includes(':5001')) {
                          e.target.src = s.replace(':5000', ':5500').replace(':5001', ':5500')
                          return
                        }
                        // After fallback, mark as missing
                        setImgError(img)
                      }}
                      alt={img.replace(/_/g, ' ').replace('.png', '')}
                      style={{
                        width: '100%',
                        maxWidth: 300,
                        maxHeight: 180,
                        objectFit: 'cover',
                        borderRadius: 12,
                        boxShadow: '0 2px 8px #0002',
                        background: '#eaeaea',
                        marginBottom: 8,
                        filter: 'drop-shadow(0 2px 8px #0001)',
                        transition: 'box-shadow 0.2s',
                      }}
                      
                    />
                    <div style={{ fontSize: 14, color: '#666', marginTop: 4, fontWeight: 500 }}>
                      {img.replace(/_/g, ' ').replace('.png', '')}
                    </div>
                    {imgError === img && <div style={{ color: 'red' }}>Image not found</div>}
                  </div>
                ))}
              </div>

              {/* Modal for full-size image */}
              {modalImg && (
                <div
                  style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    width: '100vw',
                    height: '100vh',
                    background: 'rgba(0,0,0,0.65)',
                    zIndex: 1000,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                  onClick={() => setModalImg(null)}
                >
                  <img
                    src={`http://localhost:5500/output-image?filename=${modalImg}`}
                    alt={modalImg.replace(/_/g, ' ').replace('.png', '')}
                    style={{
                      maxWidth: '90vw',
                      maxHeight: '85vh',
                      borderRadius: 16,
                      boxShadow: '0 8px 32px 0 rgba(0,0,0,0.25)',
                      background: '#fff',
                      padding: 8,
                    }}
                    onClick={e => e.stopPropagation()}
                    onError={(e) => {
                      const s = e?.target?.src || ''
                      if (s.includes(':5000') || s.includes(':5001')) {
                        e.target.src = s.replace(':5000', ':5500').replace(':5001', ':5500')
                        return
                      }
                      setImgError(modalImg)
                    }}
                  />
                  <button
                    style={{
                      position: 'fixed',
                      top: 24,
                      right: 36,
                      fontSize: 32,
                      color: '#fff',
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      zIndex: 1001,
                    }}
                    onClick={() => setModalImg(null)}
                    aria-label="Close image preview"
                  >
                    &times;
                  </button>
                </div>
              )}
            </div>
          </div>

          <div className="characteristics-column">
            <div className="characteristics-card">
              <h2 className="characteristics-title-main">Characteristics</h2>
              <div className="characteristics-grid">
                {characteristics.map((char, index) => (
                  <CharacteristicItem 
                    key={index}
                    name={char.name} 
                    value={char.value} 
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="planet-container">
        <div className="planet-image" aria-hidden="true"></div>
      </div>
    </div>
  )
})

Characteristics.displayName = 'Characteristics'

export default Characteristics
