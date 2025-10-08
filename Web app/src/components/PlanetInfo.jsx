import React, { useCallback, memo } from 'react'
import { GoogleGenerativeAI } from '@google/generative-ai';
import { useNavigate } from 'react-router-dom'
  import { usePlanet } from '../context/PlanetContext'

// Loading component
const LoadingSpinner = memo(() => (
  <div className="loading-container">
    <div className="loading-spinner"></div>
    <p>Loading planets...</p>
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

// Planet buttons component
const PlanetButtons = memo(({ 
  onNavigateToCharacteristics, 
  onPrevPlanet, 
  onNextPlanet, 
  canGoPrev, 
  canGoNext 
}) => (
  <div className="planet-buttons">
    <button 
      className="planet-btn primary" 
      onClick={onNavigateToCharacteristics}
      aria-label="View planet characteristics"
    >
      View Characteristics
    </button>
    <button 
      className="planet-btn secondary" 
      onClick={onPrevPlanet} 
      disabled={!canGoPrev}
      aria-label="Go to previous planet"
    >
      Previous Planet
    </button>
    <button 
      className="planet-btn secondary" 
      onClick={onNextPlanet} 
      disabled={!canGoNext}
      aria-label="Go to next planet"
    >
      Next Planet
    </button>
  </div>
))



async function getExoplanetDescription(planetName, apiKey) {
  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
  const prompt = `Provide a brief, 50-word description of the exoplanet named ${planetName}.`;
  try {
    const result = await model.generateContent(prompt);
    // result.response.text() returns the generated text
    return result.response.text().trim() || 'No description was generated.';
  } catch (error) {
    console.error('Error fetching exoplanet description:', error);
    return 'Could not retrieve description due to an error.';
  }
}





const PlanetInfo = memo(() => {
  const navigate = useNavigate();
  const {
    currentPlanet,
    loading,
    error,
    handleNextPlanet,
    handlePrevPlanet,
    canGoNext,
    canGoPrev,
    hasPlanets
  } = usePlanet();

  const [geminiDescription, setGeminiDescription] = React.useState('');
  const [descLoading, setDescLoading] = React.useState(false);

  const [descError, setDescError] = React.useState(null);

  const navigateToCharacteristics = useCallback(() => {
    if (hasPlanets) {
      navigate('/characteristics');
    }
  }, [navigate, hasPlanets]);

  const handleRetry = useCallback(() => {
    window.location.reload();
  }, []);

  React.useEffect(() => {
    if (!currentPlanet?.object_id) return;
    setGeminiDescription('');
    setDescError(null);
    setDescLoading(true);
    const apiKey = import.meta.env.VITE_GEMINI_KEY;
    getExoplanetDescription(currentPlanet.object_id, apiKey)
      .then(desc => {
        setGeminiDescription(desc);
        // Set the description in currentPlanet so the rest of the app can use it
        if (desc && desc !== 'No description was generated.' && desc !== 'Could not retrieve description due to an error.') {
          currentPlanet.description = desc;
        }
      })
      .catch(e => setDescError('Failed to fetch description.'))
      .finally(() => setDescLoading(false));
  }, [currentPlanet]);

  if (loading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return <ErrorMessage error={error} onRetry={handleRetry} />;
  }

  if (!currentPlanet || !hasPlanets) {
    return (
      <div className="no-data-container">
        <div className="no-data-icon">🌌</div>
        <h3>No planets available</h3>
        <p>There are no planets to display at the moment.</p>
      </div>
    );
  }



  return (
    <div className="planet-info" style={{ minHeight: '100vh' }}>
      <div className="planet-info-content">
        <h1 className="planet-title">{currentPlanet.object_id || 'Unknown Planet'}</h1>
        <p className="planet-description">
          {descLoading && 'Loading description...'}
          {descError && <span style={{color:'red'}}>{descError}</span>}
          {!descLoading && !descError && (geminiDescription || currentPlanet.description || 'No description available for this planet.')}
        </p>
        <PlanetButtons
          onNavigateToCharacteristics={navigateToCharacteristics}
          onPrevPlanet={handlePrevPlanet}
          onNextPlanet={handleNextPlanet}
          canGoPrev={canGoPrev}
          canGoNext={canGoNext}
        />
      </div>
    </div>
  );
});

PlanetInfo.displayName = 'PlanetInfo'

export default PlanetInfo