import React, { useState } from 'react';
import './App.css';

function App() {
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);

  const API_URL = 'http://localhost:5001';

  // Drag & drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files.length > 0) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleFileSelect = (file) => {
    if (!file.name.toLowerCase().endsWith('.wav')) {
      setError('Only .wav files are supported');
      return;
    }

    setUploadedFile(file);
    setError(null);
    setResults(null);
  };

  const analyzeAudio = async () => {
    if (!uploadedFile) {
      setError('Choose a file first');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analyze faled');
      }

      const data = await response.json();

      if (data.success) {
        setResults(data);
      } else {
        setError(data.error || 'Unkown error occurred');
      }
    } catch (err) {
      setError(`Virhe: ${err.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>Voice recognition</h1>
        <p></p>
      </header>

      <main className="app-main">
        {/* Upload Area */}
        <div
          className={`upload-area ${isDragging ? 'dragging' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="upload-content">
            <div className="upload-icon">📁</div>
            <h2>Drop .wav here</h2>
          </div>
        </div>

        {/* Selected File Info */}
        {uploadedFile && (
          <div className="file-info">
            <p><strong>Uploaded file:</strong> {uploadedFile.name}</p>
            <p><strong>Size:</strong> {(uploadedFile.size / 1024).toFixed(2)} KB</p>
            <button
              onClick={analyzeAudio}
              disabled={isAnalyzing}
              className="analyze-button"
            >
              {isAnalyzing ? 'Analyzing...' : 'Analyze'}
            </button>
            <button
              onClick={() => {
                setUploadedFile(null);
                setResults(null);
                setError(null);
              }}
              className="clear-button"
            >
              Clear
            </button>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="error-message">
            <strong>Error</strong> {error}
          </div>
        )}

        {/* Results */}
        {results && (
          <div className="results-container">
            <div className="result-header">
              <h2>Analyzed result</h2>
            </div>

            <div className="detected-language">
              <h3>Detected language</h3>
              <div className="language-badge">
                {results.detected_language}
              </div>
              <p className="confidence-text">
                Confidence: <strong>{(results.confidence * 100).toFixed(2)}%</strong>
              </p>
            </div>

            {/* Confidence Chart */}
            <div className="chart-container">
              <h3>Confidence for all languages</h3>
              <div className="confidence-bars">
                {Object.entries(results.all_confidences).map(([language, conf]) => (
                  <div key={language} className="confidence-bar-item">
                    <label>{language}</label>
                    <div className="bar-container">
                      <div
                        className={`bar ${
                          language === results.detected_language ? 'primary' : 'secondary'
                        }`}
                        style={{ width: `${conf * 100}%` }}
                      />
                    </div>
                    <span className="bar-value">{(conf * 100).toFixed(2)}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Audio Info */}
            <div className="audio-info">
              <h3>Audio file results</h3>
              <p><strong>Duration:</strong> {results.audio_duration.toFixed(2)}s</p>
              <p><strong>Sample frequency:</strong> {results.sample_rate} Hz</p>
            </div>

            {/* Analyze Another Button */}
            <button
              onClick={() => {
                setUploadedFile(null);
                setResults(null);
                setError(null);
              }}
              className="analyze-another-button"
            >
              Analyze another file
            </button>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>Powered by TensorFlow AI Model</p>
      </footer>
    </div>
  );
}

export default App;
