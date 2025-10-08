import React, { useState, useCallback, memo } from 'react'
import { usePlanet } from '../context/PlanetContext'

// Menu toggle button component
const MenuToggleButton = memo(({ isOpen, onToggle }) => (
  <button 
    className="menu-toggle-btn" 
    onClick={onToggle} 
    aria-label={isOpen ? "Close planet menu" : "Open planet menu"}
    aria-expanded={isOpen}
  >
    <span className={`hamburger ${isOpen ? 'active' : ''}`}>
      <span></span>
      <span></span>
      <span></span>
    </span>
  </button>
))

// Planet card component
const PlanetCard = memo(({ planet, index }) => (
  <div 
    className="planet-card"
    role="button"
    tabIndex={0}
    aria-label={`View ${planet.object_id || 'Unknown Planet'} details`}
  >
    <div 
      className="planet-image" 
      style={{ backgroundImage: `url('/planet_svg.svg')` }}
      aria-hidden="true"
    ></div>
    <div className="planet-card-name">{planet.object_id || 'Unknown Planet'}</div>
  </div>
))

// Filter group component
const FilterGroup = memo(({ title, options }) => (
  <div className="filter-group">
    <div className="filter-title">{title}</div>
    <div className="filter-options">
      {options.map((option, index) => (
        <button 
          key={index}
          className={`filter-option ${option.active ? 'active' : ''}`}
          aria-pressed={option.active}
        >
          {option.label}
        </button>
      ))}
    </div>
  </div>
))

const Header = memo(() => {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const { planets, loading, generateSafeKey } = usePlanet()

  const toggleMenu = useCallback(() => {
    setIsMenuOpen(prev => !prev)
  }, [])

  const closeMenu = useCallback(() => {
    setIsMenuOpen(false)
  }, [])

  // Filter options data
  const filterOptions = {
    sunGroup: [
      { label: 'Yes', active: true },
      { label: 'No', active: false }
    ],
    massRegime: [
      { label: 'Super-Earth', active: true },
      { label: 'Giant planet', active: false },
      { label: 'Ice giant', active: false },
      { label: 'Mini-Neptune', active: false },
      { label: 'Super-Jupiter', active: false }
    ],
    orbitalRegime: [
      { label: 'Circumbinary planet', active: false },
      { label: 'Double planet', active: false },
      { label: 'Eccentric Jupiter', active: false },
      { label: 'Exoplanet', active: false }
    ]
  }

  return (
    <>
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
          
          <div className="header-right">
            <MenuToggleButton isOpen={isMenuOpen} onToggle={toggleMenu} />
            <button
              className="donate-btn"
              aria-label="Redirects to donation page"
              onClick={() => window.location.href = "https://deddee362911.ngrok-free.app"}
            >
              Predict
            </button>
          </div>
        </div>
      </header>

      {/* Full-screen sidebar overlay */}
      <div 
        className={`sidebar-overlay ${isMenuOpen ? 'active' : ''}`} 
        onClick={closeMenu}
        role="dialog"
        aria-modal="true"
        aria-hidden={!isMenuOpen}
      >
        <div 
          className={`sidebar-content ${isMenuOpen ? 'active' : ''}`} 
          onClick={(e) => e.stopPropagation()}
        >
          <div className="sidebar-header">
            <button 
              className="close-btn" 
              onClick={closeMenu} 
              aria-label="Close menu"
            >
              <span className="close-icon" aria-hidden="true">&times;</span>
            </button>
          </div>
          
          <div className="sidebar-title">Filters</div>
          
          <div className="filters-section">
            <FilterGroup title="Sun Group" options={filterOptions.sunGroup} />
            <FilterGroup title="By mass regime" options={filterOptions.massRegime} />
            <FilterGroup title="By orbital regime" options={filterOptions.orbitalRegime} />
          </div>

          <div className="planets-grid-header">Planets</div>
          <div className="planets-grid" role="grid" aria-label="Available planets">
            {loading ? (
              <div className="loading-planets">Loading planets...</div>
            ) : planets.length > 0 ? (
              planets.map((planet, index) => (
                <PlanetCard 
                  key={generateSafeKey(planet, index)} 
                  planet={planet} 
                  index={index} 
                />
              ))
            ) : (
              <div className="no-planets">No planets available</div>
            )}
          </div>
        </div>
      </div>
    </>
  )
})

Header.displayName = 'Header'

export default Header

