import React, { useState } from 'react';

function ControlPanel({
  isRecording,
  onStart,
  onStop,
  inputDevices,
  selectedInputDevice,
  onSelectInputDevice,
  onFileUpload,
  downloadLinks,
}) {
  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (file && onFileUpload) {
      onFileUpload(file);
    }
  };

  return (
    <div className="control-panel">
      <div className="control-grid">
        <section className="control-group">
          <h4>Live Input</h4>
          <div className="record-controls">
            <div className="device-picker">
              <label htmlFor="input-device-select">Input device</label>
              <select
                id="input-device-select"
                value={selectedInputDevice}
                onChange={(e) => onSelectInputDevice?.(e.target.value)}
                disabled={isRecording}
              >
                <option value="">System default</option>
                {(inputDevices || []).map((device) => (
                  <option key={device.deviceId} value={device.deviceId}>
                    {device.label}
                  </option>
                ))}
              </select>
            </div>

            <button
              className={`btn ${isRecording ? 'danger' : 'primary'}`}
              onClick={isRecording ? onStop : onStart}
            >
              {isRecording ? 'Stop Live Processing' : 'Start Live Processing'}
            </button>
          </div>
        </section>

        <section className="control-group">
          <h4>File Input</h4>
          <div className="upload-section">
            <label htmlFor="file-upload" className="btn secondary">
              Upload Audio File
            </label>
            <input
              id="file-upload"
              type="file"
              accept=".wav,.mp3,.flac,.m4a"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
          </div>
        </section>

        <section className="control-group">
          <h4>Audio Comparison</h4>
          {downloadLinks.cleanedAudio ? (
            <div className="downloads compact-downloads">
              <div className="audio-compare-grid">
                <div className="audio-card">
                  <p className="audio-label">Original Audio</p>
                  <audio controls preload="metadata" src={downloadLinks.originalAudio} className="audio-player">
                    Your browser does not support audio playback.
                  </audio>
                </div>
                <div className="audio-card">
                  <p className="audio-label">Cleaned Audio</p>
                  <audio controls preload="metadata" src={downloadLinks.cleanedAudio} className="audio-player">
                    Your browser does not support audio playback.
                  </audio>
                </div>
              </div>

              <div className="download-links-wrap">
                {downloadLinks.originalAudio && (
                  <a href={downloadLinks.originalAudio} className="download-link">Download Original Audio (.wav)</a>
                )}
                <a href={downloadLinks.cleanedAudio} className="download-link">Download Cleaned Audio (.wav)</a>
                <a href={downloadLinks.transcript} className="download-link">Download Refined Transcript (.txt)</a>
              </div>
            </div>
          ) : (
            <p className="placeholder">Run live processing or upload a file to unlock A/B audio playback.</p>
          )}
        </section>
      </div>
    </div>
  );
}

export default ControlPanel;
