import React from 'react';

function TranscriptPanel({ title, text, variant = 'original' }) {
  return (
    <div className={`transcript-panel ${variant}`}>
      <div className="transcript-head">
        <h3>{title}</h3>
        <span className="transcript-badge">{variant === 'refined' ? 'Polished' : 'Raw'}</span>
      </div>
      <div className="transcript-content-wrap">
        <div className="transcript-content">
          {text || <span className="placeholder">No transcript yet...</span>}
        </div>
      </div>
    </div>
  );
}

export default TranscriptPanel;
