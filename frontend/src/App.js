import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [mode, setMode] = useState('single'); // 'single' or 'batch'
  const [batchResults, setBatchResults] = useState(null);
  const [batchSummary, setBatchSummary] = useState(null);
  const [isLoadingBatch, setIsLoadingBatch] = useState(false);

  const fileInputRef = useRef(null);
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
    if (files.length > 0) handleFileSelect(files[0]);
  };

  // File input handler
  const handleFileInput = (e) => {
    if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
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
      setError('Drop a file first');
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

      if (!response.ok) throw new Error('Analysis failed');

      const data = await response.json();
      if (data.success) setResults(data);
      else setError(data.error || 'Unknown error occurred');
    } catch (err) {
      setError(`${err.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const runBatchTest = async () => {
    setIsLoadingBatch(true);
    setError(null);
    setBatchResults(null);
    setBatchSummary(null);

    try {
      const response = await fetch(`${API_URL}/test-batch`, {
        method: 'GET',
      });

      if (!response.ok) throw new Error('Batch test failed');

      const data = await response.json();
      if (data.success) {
        setBatchResults(data.results);
        setBatchSummary({
          total: data.total_files,
          correct: data.correct_predictions,
          accuracy: data.accuracy
        });
      } else {
        setError(data.error || 'Batch test failed');
      }
    } catch (err) {
      setError(`Batch test error: ${err.message}`);
    } finally {
      setIsLoadingBatch(false);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>Voice recognition</h1>
      </header>

      <main className="app-main">
        {/* Mode Selector */}
        <div className="mode-selector">
          <button
            className={`mode-button ${mode === 'single' ? 'active' : ''}`}
            onClick={() => {
              setMode('single');
              setBatchResults(null);
              setBatchSummary(null);
            }}
          >
            Single File Test
          </button>
          <button
            className={`mode-button ${mode === 'batch' ? 'active' : ''}`}
            onClick={() => {
              setMode('batch');
              setResults(null);
              setUploadedFile(null);
            }}
          >
            Batch Test (Console Demo)
          </button>
        </div>

        {/* Single File Mode */}
        {mode === 'single' && (
          <>
            {/* Hidden file input */}
            <input
              type="file"
              accept=".wav"
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={handleFileInput}
            />

            {/* Upload Area */}
            <div
              className={`upload-area ${isDragging ? 'dragging' : ''}`}
              onClick={() => fileInputRef.current.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="upload-content">
                <div className="upload-icon">📁</div>
                <h2>Drop .wav here or click to select</h2>
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

            {/* Results */}
            {results && (
              <div className="results-container">
                <div className="result-header">
                  <h2>Analyzed result</h2>
                  {results.true_language && (
                    <div className={`correctness-badge ${results.is_correct ? 'correct' : 'wrong'}`}>
                      {results.is_correct ? '✓ CORRECT' : '✗ WRONG'}
                    </div>
                  )}
                </div>

                <div className="detected-language">
                  <h3>Detected language</h3>
                  <div className="language-badge">
                    {results.detected_language}
                  </div>
                  <p className="confidence-text">
                    Confidence: <strong>{(results.confidence * 100).toFixed(2)}%</strong>
                  </p>
                  {results.true_language && (
                    <p className="true-language">
                      Expected: <strong>{results.true_language}</strong>
                    </p>
                  )}
                </div>

                <div className="chart-container">
                  <h3>Confidence for all languages</h3>
                  <div className="confidence-bars">
                    {Object.entries(results.all_confidences).map(([language, conf]) => (
                      <div key={language} className="confidence-bar-item">
                        <label>{language}</label>
                        <div className="bar-container">
                          <div
                            className={`bar ${language === results.detected_language ? 'primary' : 'secondary'}`}
                            style={{ width: `${conf * 100}%` }}
                          />
                        </div>
                        <span className="bar-value">{(conf * 100).toFixed(2)}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="audio-info">
                  <h3>Audio file results</h3>
                  <p><strong>Duration:</strong> {results.audio_duration.toFixed(2)}s</p>
                  <p><strong>Sample frequency:</strong> {results.sample_rate} Hz</p>
                </div>

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
          </>
        )}

        {/* Batch Test Mode */}
        {mode === 'batch' && (
          <>
            <div className="batch-test-container">
              <div className="console-info">
                <h2>Console Batch Test</h2>
                <p>This replicates the <code>main.py</code> console script behavior</p>
                <p>Tests all 40 files in <code>data/test/</code> directory</p>
              </div>

              <button
                onClick={runBatchTest}
                disabled={isLoadingBatch}
                className="batch-test-button"
              >
                {isLoadingBatch ? 'Running Tests...' : 'Run Batch Test'}
              </button>
            </div>

            {/* Batch Results Display */}
            {batchResults && batchSummary && (
              <div className="console-results">
                <div className="console-header">
                  <h2>Batch Test Results</h2>
                  <button
                    onClick={() => {
                      setBatchResults(null);
                      setBatchSummary(null);
                    }}
                    className="clear-button"
                  >
                    Clear Results
                  </button>
                </div>

                {/* Summary Box */}
                <div className="summary-box">
                  <h3>SUMMARY</h3>
                  <div className={`accuracy-display ${
                    batchSummary.accuracy >= 80 ? 'excellent' :
                    batchSummary.accuracy >= 60 ? 'good' : 'poor'
                  }`}>
                    <span className="accuracy-label">Accuracy:</span>
                    <span className="accuracy-value">
                      {batchSummary.correct}/{batchSummary.total}
                      ({batchSummary.accuracy.toFixed(1)}%)
                    </span>
                  </div>
                </div>

                {/* File-by-file results */}
                <div className="console-output">
                  <h3>Test Results by File</h3>
                  <div className="results-table">
                    {batchResults.map((result, idx) => (
                      <div
                        key={idx}
                        className={`result-row ${result.is_correct ? 'correct' : 'incorrect'}`}
                      >
                        <div className="result-filename">{result.filename}</div>
                        <div className="result-prediction">
                          Prediction: <strong>{result.predicted_language}</strong>
                          ({(result.confidence * 100).toFixed(1)}%)
                        </div>
                        <div className={`result-status ${result.is_correct ? 'correct' : 'wrong'}`}>
                          {result.is_correct ? '✓ CORRECT' : '✗ WRONG'}
                        </div>
                        <div className="result-true">
                          True: {result.true_language}
                        </div>
                        <div className="result-transcript">
                          {result.transcript.substring(0, 50)}
                          {result.transcript.length > 50 ? '...' : ''}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* Error Message */}
        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
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
