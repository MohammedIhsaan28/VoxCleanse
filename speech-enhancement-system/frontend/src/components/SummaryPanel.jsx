import React from 'react';

function SummaryPanel({
  summary,
  onSummarize,
  isSummarizing,
  sourceWordCount,
  summaryWordCount,
  error,
  disabled,
}) {
  return (
    <div className="summary-panel card-lite">
      <div className="summary-head">
        <div>
          <h3>Transcript Summary</h3>
          <p>Generate a concise version of the refined transcript for quick review.</p>
        </div>
        <button className="btn accent summary-btn" onClick={onSummarize} disabled={disabled || isSummarizing}>
          {isSummarizing ? 'Summarizing...' : 'Summarize Transcript'}
        </button>
      </div>

      <div className="summary-meta">
        {sourceWordCount > 0 && <span>{sourceWordCount} source words</span>}
        {summaryWordCount > 0 && <span>{summaryWordCount} summary words</span>}
      </div>

      <div className="summary-output-wrap">
        {summary ? (
          <p className="summary-output">{summary}</p>
        ) : (
          <span className="placeholder">No summary yet. Click summarize to generate one.</span>
        )}
      </div>

      {error && <p className="error-text summary-error">{error}</p>}
    </div>
  );
}

export default SummaryPanel;
