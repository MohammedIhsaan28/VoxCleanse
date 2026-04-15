import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ControlPanel from './components/ControlPanel';
import TranscriptPanel from './components/TranscriptPanel';
import SummaryPanel from './components/SummaryPanel';
import AudioStreamer from './services/audioStreamer';
import WebSocketService from './services/websocketService';

const API_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/audio';

function TinySparkline({ values, width = 180, height = 44 }) {
  const points = Array.isArray(values) ? values.filter((v) => Number.isFinite(Number(v))) : [];
  if (points.length === 0) {
    return <div className="sparkline-empty">No trend yet</div>;
  }

  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = Math.max(1, max - min);
  const stepX = points.length > 1 ? width / (points.length - 1) : width;

  const linePath = points
    .map((value, idx) => {
      const x = idx * stepX;
      const y = height - ((Number(value) - min) / range) * height;
      return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(' ');

  return (
    <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
      <path d={linePath} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function ScoreRing({ label, value, level, accentClass = '', centerLines = null }) {
  const safeValue = Math.max(0, Math.min(100, Number(value || 0)));
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference - (safeValue / 100) * circumference;

  return (
    <div className={`score-ring ${accentClass}`}>
      <svg viewBox="0 0 100 100" className="score-ring-svg" aria-hidden="true">
        <circle className="score-ring-track" cx="50" cy="50" r={radius} />
        <circle
          className="score-ring-fill"
          cx="50"
          cy="50"
          r={radius}
          style={{ strokeDasharray: circumference, strokeDashoffset: dashOffset }}
        />
      </svg>
      <div className="score-ring-content">
        {centerLines ? (
          <>
            <strong>{centerLines[0]}</strong>
            <span>{centerLines[1]}</span>
            <em>{centerLines[2]}</em>
          </>
        ) : (
          <>
            <strong>{safeValue.toFixed(0)}</strong>
            <span>{label}</span>
            {level && <em>{level}</em>}
          </>
        )}
      </div>
    </div>
  );
}

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [originalTranscript, setOriginalTranscript] = useState('');
  const [refinedTranscript, setRefinedTranscript] = useState('');
  const [customFilters, setCustomFilters] = useState({});
  const [downloadLinks, setDownloadLinks] = useState({ audio: '', originalAudio: '', cleanedAudio: '', transcript: '' });
  const [sessionId, setSessionId] = useState('');
  const [scores, setScores] = useState(null);
  const [scoreHistory, setScoreHistory] = useState([]);
  const [savedSessions, setSavedSessions] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [journalRange, setJournalRange] = useState('7d');
  const [journalMode, setJournalMode] = useState('all');
  const [journalSort, setJournalSort] = useState('latest');
  const [compareSessionIds, setCompareSessionIds] = useState([]);
  const [error, setError] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [inputDevices, setInputDevices] = useState([]);
  const [selectedInputDevice, setSelectedInputDevice] = useState('');
  const [summaryText, setSummaryText] = useState('');
  const [summaryError, setSummaryError] = useState('');
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [summarySourceWordCount, setSummarySourceWordCount] = useState(0);
  const [summaryWordCount, setSummaryWordCount] = useState(0);

  const wsRef = useRef(null);
  const audioStreamerRef = useRef(null);
  const audioPlaybackContextRef = useRef(null);

  const fetchPersistedHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const response = await fetch(`${API_BASE}/sessions/history?limit=40`);
      if (!response.ok) {
        throw new Error(`History API returned ${response.status}`);
      }
      const payload = await response.json();
      const items = Array.isArray(payload.items) ? payload.items : [];
      setSavedSessions(items);
    } catch (_e) {
      // Keep silent to avoid disrupting live UX when history API is unavailable.
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const handleWsMessage = useCallback((message) => {
    if (message.type === 'session_started') {
      setSessionId(message.session_id);
      setConnectionStatus('connected');
      setScoreHistory([]);
      setScores(null);
      return;
    }

    if (message.type === 'processed') {
      if (message.original_text) {
        setOriginalTranscript((prev) => `${prev} ${message.original_text}`.trim());
      }

      if (message.refined_text) {
        setRefinedTranscript((prev) => `${prev} ${message.refined_text}`.trim());
      }

      if (message.rolling_scores) {
        setScores(message.rolling_scores);
      }
      if (message.score_history_point) {
        setScoreHistory((prev) => [...prev, message.score_history_point]);
      }

      // Optional: play cleaned audio chunk
      if (message.cleaned_audio && message.cleaned_audio.length > 0) {
        playAudioChunk(new Float32Array(message.cleaned_audio));
      }
      return;
    }

    if (message.type === 'config_updated') {
      if (typeof message.refined_transcript === 'string') {
        setRefinedTranscript(message.refined_transcript);
      }
      return;
    }

    if (message.type === 'score_update') {
      if (message.scores) {
        setScores(message.scores);
      }
      if (message.history_point) {
        setScoreHistory((prev) => [...prev, message.history_point]);
      }
      return;
    }

    if (message.type === 'session_ended') {
      const originalAudioPath = message.download_urls.original_audio || '';
      const cleanedAudioPath = message.download_urls.cleaned_audio || message.download_urls.audio || '';
      setDownloadLinks({
        audio: message.download_urls.audio ? `${API_BASE}${message.download_urls.audio}` : '',
        originalAudio: originalAudioPath ? `${API_BASE}${originalAudioPath}` : '',
        cleanedAudio: cleanedAudioPath ? `${API_BASE}${cleanedAudioPath}` : '',
        transcript: `${API_BASE}${message.download_urls.transcript}`,
      });
      setScores(message.scores || null);
      if (Array.isArray(message.score_history)) {
        setScoreHistory(message.score_history);
      }
      setConnectionStatus('disconnected');
      void fetchPersistedHistory();
      return;
    }

    if (message.type === 'error') {
      setError(message.message || 'Unknown WebSocket error');
    }
  }, [fetchPersistedHistory]);

  const handleWsError = useCallback(() => {
    setError('WebSocket connection error');
    setConnectionStatus('error');
  }, []);

  const handleWsClose = useCallback(() => {
    setConnectionStatus('disconnected');
  }, []);

  const connectWebSocket = useCallback(async () => {
    setConnectionStatus('connecting');
    const wsService = new WebSocketService(WS_URL, handleWsMessage, handleWsError, handleWsClose);
    await wsService.connect();
    wsRef.current = wsService;

    // Send current custom filters
    if (Object.keys(customFilters).length > 0) {
      wsService.sendConfig(customFilters);
    }

    return wsService;
  }, [customFilters, handleWsClose, handleWsError, handleWsMessage]);

  const playAudioChunk = (audioChunk) => {
    try {
      if (!audioPlaybackContextRef.current) {
        audioPlaybackContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }

      const context = audioPlaybackContextRef.current;
      const buffer = context.createBuffer(1, audioChunk.length, 16000);
      buffer.copyToChannel(audioChunk, 0);

      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      source.start();
    } catch (e) {
      // Ignore playback errors to avoid interrupting processing
      console.warn('Audio playback warning:', e);
    }
  };

  const startRecording = async () => {
    try {
      setError('');
      setOriginalTranscript('');
      setRefinedTranscript('');
      setDownloadLinks({ audio: '', originalAudio: '', cleanedAudio: '', transcript: '' });
      setScores(null);
      setScoreHistory([]);
      setSummaryText('');
      setSummaryError('');
      setSummarySourceWordCount(0);
      setSummaryWordCount(0);

      const wsService = await connectWebSocket();

      const streamer = new AudioStreamer((audioChunk) => {
        if (wsService.isConnected()) {
          wsService.sendAudioChunk(audioChunk, 16000);
        }
      });

      await streamer.start(selectedInputDevice);
      audioStreamerRef.current = streamer;
      setIsRecording(true);

      // Refresh labels after mic permission is granted.
      const devices = await AudioStreamer.getInputDevices();
      setInputDevices(devices);
    } catch (e) {
      setError(`Failed to start recording: ${e.message}`);
      setIsRecording(false);
      setConnectionStatus('error');
    }
  };

  const stopRecording = () => {
    try {
      if (audioStreamerRef.current) {
        audioStreamerRef.current.stop();
        audioStreamerRef.current = null;
      }

      if (wsRef.current && wsRef.current.isConnected()) {
        wsRef.current.endSession();
        // Keep WS open briefly to receive final message
        setTimeout(() => {
          if (wsRef.current) {
            wsRef.current.disconnect();
            wsRef.current = null;
          }
        }, 1000);
      }

      setIsRecording(false);
    } catch (e) {
      setError(`Failed to stop recording: ${e.message}`);
    }
  };

  const handleUpdateFilter = (word, replacement) => {
    const updated = {
      ...customFilters,
      [word]: replacement,
    };

    setCustomFilters(updated);

    if (wsRef.current && wsRef.current.isConnected()) {
      wsRef.current.sendConfig(updated);
    }
  };

  const handleRemoveFilter = (word) => {
    const updated = { ...customFilters };
    delete updated[word];
    setCustomFilters(updated);

    if (wsRef.current && wsRef.current.isConnected()) {
      wsRef.current.sendConfig(updated);
    }
  };

  const handleFileUpload = async (file) => {
    setIsUploading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('custom_filters', JSON.stringify(customFilters));

      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();

      setSessionId(result.session_id);
      setOriginalTranscript(result.original_transcript || '');
      setRefinedTranscript(result.refined_transcript || '');
      setScores(result.scores || null);
      setScoreHistory(Array.isArray(result.score_history) ? result.score_history : []);
      const originalAudioPath = result.download_urls.original_audio || '';
      const cleanedAudioPath = result.download_urls.cleaned_audio || result.download_urls.audio || '';
      setDownloadLinks({
        audio: result.download_urls.audio ? `${API_BASE}${result.download_urls.audio}` : '',
        originalAudio: originalAudioPath ? `${API_BASE}${originalAudioPath}` : '',
        cleanedAudio: cleanedAudioPath ? `${API_BASE}${cleanedAudioPath}` : '',
        transcript: `${API_BASE}${result.download_urls.transcript}`,
      });
      void fetchPersistedHistory();
    } catch (e) {
      setError(`File upload error: ${e.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSummarizeTranscript = useCallback(async () => {
    const transcriptText = refinedTranscript.trim();
    if (!transcriptText) {
      setSummaryError('Add transcript text before summarizing.');
      return;
    }

    setIsSummarizing(true);
    setSummaryError('');

    try {
      const response = await fetch(`${API_BASE}/summarize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: transcriptText }),
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `Summarization failed: ${response.statusText}`);
      }

      const result = await response.json();
      setSummaryText(result.summary || '');
      setSummarySourceWordCount(result.source_word_count || 0);
      setSummaryWordCount(result.summary_word_count || 0);
    } catch (e) {
      setSummaryError(e.message || 'Unable to summarize transcript');
      setSummaryText('');
      setSummarySourceWordCount(0);
      setSummaryWordCount(0);
    } finally {
      setIsSummarizing(false);
    }
  }, [refinedTranscript]);

  useEffect(() => {
    AudioStreamer.getInputDevices()
      .then((devices) => setInputDevices(devices))
      .catch(() => setInputDevices([]));

    void fetchPersistedHistory();

    return () => {
      if (audioStreamerRef.current) {
        audioStreamerRef.current.stop();
      }
      if (wsRef.current) {
        wsRef.current.disconnect();
      }
      if (audioPlaybackContextRef.current) {
        audioPlaybackContextRef.current.close();
      }
    };
  }, [fetchPersistedHistory]);

  const statusLabel = useMemo(() => {
    if (isUploading) return 'Processing uploaded file...';
    return `Status: ${connectionStatus}`;
  }, [connectionStatus, isUploading]);

  const scoreBars = useMemo(() => {
    const safe = (v) => Math.max(0, Math.min(100, Number(v || 0)));
    if (!scores) {
      return [];
    }
    return [
      { label: 'Fluency', value: safe(scores.fluency) },
      { label: 'Vocabulary', value: safe(scores.vocabulary) },
      { label: 'Communication', value: safe(scores.communication) },
      { label: 'Overall', value: safe(scores.overall), overall: true },
    ];
  }, [scores]);

  const scoreRings = useMemo(() => {
    if (!scores) return [];
    return [
      { label: 'Fluency', value: scores.fluency, accentClass: 'fluency' },
      { label: 'Vocabulary', value: scores.vocabulary, accentClass: 'vocabulary' },
      { label: 'Communication', value: scores.communication, accentClass: 'communication' },
    ];
  }, [scores]);

  const historyTail = useMemo(() => {
    if (!Array.isArray(scoreHistory)) return [];
    return scoreHistory.slice(-8);
  }, [scoreHistory]);

  const rollingOverallValues = useMemo(() => {
    return historyTail.map((point) => Number(point.overall || 0));
  }, [historyTail]);

  const groupedSessions = useMemo(() => {
    const now = Date.now();
    const rangeDays = journalRange === '7d' ? 7 : null;
    const cutoff = rangeDays ? now - rangeDays * 24 * 60 * 60 * 1000 : null;

    const filtered = savedSessions.filter((item) => {
      const createdAt = item.created_at ? new Date(item.created_at) : null;
      const createdTs = createdAt && !Number.isNaN(createdAt.getTime()) ? createdAt.getTime() : null;
      const modeMatches = journalMode === 'all' || item.mode === journalMode;
      const rangeMatches = cutoff === null || (createdTs !== null && createdTs >= cutoff);
      return modeMatches && rangeMatches;
    });

    const ordered = [...filtered].sort((a, b) => {
      const aTs = a.created_at ? new Date(a.created_at).getTime() : 0;
      const bTs = b.created_at ? new Date(b.created_at).getTime() : 0;
      const aScore = Number(a?.scores?.overall || 0);
      const bScore = Number(b?.scores?.overall || 0);

      if (journalSort === 'best') {
        return bScore - aScore || bTs - aTs;
      }
      return bTs - aTs || bScore - aScore;
    });

    const grouped = {};
    for (const item of ordered) {
      const dt = item.created_at ? new Date(item.created_at) : null;
      const key = dt && !Number.isNaN(dt.getTime()) ? dt.toLocaleDateString() : 'Unknown Date';
      if (!grouped[key]) {
        grouped[key] = [];
      }
      grouped[key].push(item);
    }

    return Object.entries(grouped).map(([date, items]) => {
      const avgOverall = items.reduce((acc, x) => acc + Number(x?.scores?.overall || 0), 0) / Math.max(1, items.length);
      return { date, items, avgOverall };
    });
  }, [journalMode, journalRange, journalSort, savedSessions]);

  const compareSessions = useMemo(() => {
    const lookup = new Map(savedSessions.map((item) => [item.session_id, item]));
    return compareSessionIds.map((sessionId) => lookup.get(sessionId)).filter(Boolean);
  }, [compareSessionIds, savedSessions]);

  const compareSummary = useMemo(() => {
    if (compareSessions.length < 2) return null;
    const total = compareSessions.length;
    const sum = compareSessions.reduce(
      (acc, item) => {
        acc.fluency += Number(item?.scores?.fluency || 0);
        acc.vocabulary += Number(item?.scores?.vocabulary || 0);
        acc.communication += Number(item?.scores?.communication || 0);
        acc.overall += Number(item?.scores?.overall || 0);
        return acc;
      },
      { fluency: 0, vocabulary: 0, communication: 0, overall: 0 }
    );
    return {
      fluency: sum.fluency / total,
      vocabulary: sum.vocabulary / total,
      communication: sum.communication / total,
      overall: sum.overall / total,
    };
  }, [compareSessions]);

  const toggleCompareSession = useCallback((sessionId) => {
    setCompareSessionIds((prev) => {
      if (prev.includes(sessionId)) {
        return prev.filter((id) => id !== sessionId);
      }
      if (prev.length >= 3) {
        return [...prev.slice(1), sessionId];
      }
      return [...prev, sessionId];
    });
  }, []);

  const exportCompareSessions = useCallback((format) => {
    if (compareSessions.length === 0) return;

    const payload = compareSessions.map((item) => ({
      session_id: item.session_id,
      mode: item.mode || 'session',
      created_at: item.created_at || '',
      fluency: Number(item?.scores?.fluency || 0),
      vocabulary: Number(item?.scores?.vocabulary || 0),
      communication: Number(item?.scores?.communication || 0),
      overall: Number(item?.scores?.overall || 0),
      wpm: Number(item?.metrics?.wpm || 0),
      total_words: Number(item?.metrics?.total_words || 0),
      feedback: Array.isArray(item.feedback) ? item.feedback : item.feedback || item?.feedback || [],
    }));

    let blob;
    let filename;

    if (format === 'csv') {
      const header = [
        'session_id',
        'mode',
        'created_at',
        'fluency',
        'vocabulary',
        'communication',
        'overall',
        'wpm',
        'total_words',
      ];
      const csvRows = [
        header.join(','),
        ...payload.map((row) =>
          [
            row.session_id,
            row.mode,
            row.created_at,
            row.fluency,
            row.vocabulary,
            row.communication,
            row.overall,
            row.wpm,
            row.total_words,
          ]
            .map((value) => `"${String(value).replaceAll('"', '""')}"`)
            .join(',')
        ),
      ];
      blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8' });
      filename = `compare-sessions-${Date.now()}.csv`;
    } else {
      blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json;charset=utf-8' });
      filename = `compare-sessions-${Date.now()}.json`;
    }

    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }, [compareSessions]);

  const overallDelta = useMemo(() => {
    if (savedSessions.length < 2) return null;
    const latest = Number(savedSessions[0]?.scores?.overall || 0);
    const previous = Number(savedSessions[1]?.scores?.overall || 0);
    return latest - previous;
  }, [savedSessions]);

  useEffect(() => {
    setSummaryText('');
    setSummaryError('');
    setSummarySourceWordCount(0);
    setSummaryWordCount(0);
  }, [refinedTranscript]);

  return (
    <div className="app-shell">
      <div className="ambient-bg" />
      <header className="hero">
        <p className="eyebrow">AI Speech Enhancement Suite</p>
        <h1>Real-Time Voice Cleanup and Transcript Refinement</h1>
        <p className="subtitle">
          DeepFilterNet denoising + Whisper transcription + professional language polishing
        </p>
        <div className="status-chip">{statusLabel}</div>
      </header>

      <main className="grid-layout">
        <section className="left-column card">
          <ControlPanel
            isRecording={isRecording}
            onStart={startRecording}
            onStop={stopRecording}
            inputDevices={inputDevices}
            selectedInputDevice={selectedInputDevice}
            onSelectInputDevice={setSelectedInputDevice}
            customFilters={customFilters}
            onUpdateFilter={handleUpdateFilter}
            onRemoveFilter={handleRemoveFilter}
            onFileUpload={handleFileUpload}
          />

          {downloadLinks.audio && (
            <div className="downloads">
              <h4>Download Outputs</h4>
              <a href={downloadLinks.cleanedAudio || downloadLinks.audio} className="download-link">Download Cleaned Audio (.wav)</a>
              {downloadLinks.originalAudio && (
                <a href={downloadLinks.originalAudio} className="download-link">Download Original Audio (.wav)</a>
              )}
              <a href={downloadLinks.transcript} className="download-link">Download Refined Transcript (.txt)</a>
            </div>
          )}

          {scores && (
            <div className="score-card elevated">
              <div className="score-card-head">
                <h4>Communication Scorecard</h4>
                <div className="level-pill">
                  <span>{scores.communication_level || 'A1'}</span>
                  <strong>{scores.communication_level_label || 'Beginner'}</strong>
                </div>
              </div>

              <div className="score-ring-layout">
                <div className="ring-group">
                  {scoreRings.map((ring) => (
                    <ScoreRing
                      key={ring.label}
                      label={ring.label}
                      value={ring.value}
                      accentClass={ring.accentClass}
                    />
                  ))}
                </div>

                <div className="overall-ring-panel">
                  <ScoreRing
                    label="Overall"
                    value={scores.overall}
                    level={scores.communication_level || 'A1'}
                    centerLines={[
                      scores.communication_level || 'A1',
                      scores.communication_level_label || 'Beginner',
                      `${Number(scores.overall || 0).toFixed(0)}/100`,
                    ]}
                    accentClass="overall"
                  />
                  <p>{scores.communication_level_description || 'Communication level summary appears here.'}</p>
                </div>
              </div>

              <div className="score-bars">
                {scoreBars.map((item) => (
                  <div key={item.label} className={`score-bar-row ${item.overall ? 'overall' : ''}`}>
                    <div className="score-bar-head">
                      <span>{item.label}</span>
                      <span>{item.value.toFixed(1)}</span>
                    </div>
                    <div className="score-bar-track">
                      <div className="score-bar-fill" style={{ width: `${item.value}%` }} />
                    </div>
                  </div>
                ))}
              </div>

              <div className="score-history">
                <p>Rolling Progress ({scoreHistory.length} points)</p>
                <TinySparkline values={rollingOverallValues} />
                {historyTail.length === 0 ? (
                  <span className="placeholder">Score history will appear during live analysis.</span>
                ) : (
                  <div className="history-list">
                    {historyTail.map((point) => (
                      <div key={`${point.step}-${point.timestamp}`} className="history-item">
                        <span className="history-step">#{point.step}</span>
                        <span className="history-overall">Overall {Number(point.overall || 0).toFixed(1)}</span>
                        <span className="history-meta">
                          F {Number(point.fluency || 0).toFixed(0)} | V {Number(point.vocabulary || 0).toFixed(0)} | C {Number(point.communication || 0).toFixed(0)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {Array.isArray(scores.feedback) && scores.feedback.length > 0 && (
                <div className="score-feedback">
                  <p>Feedback</p>
                  <ul>
                    {scores.feedback.map((line, idx) => (
                      <li key={`${idx}-${line}`}>{line}</li>
                    ))}
                  </ul>
                </div>
              )}

              {scores.overall_feedback && (
                <div className="overall-feedback">
                  <p>Overall Feedback</p>
                  <strong>{scores.overall_feedback}</strong>
                </div>
              )}

              {scores.detailed_feedback && (
                <div className="detailed-feedback-grid">
                  <div className="detail-card">
                    <p>Pace Feedback</p>
                    <ul>
                      {(scores.detailed_feedback.pace_feedback || []).map((item, idx) => (
                        <li key={`pace-${idx}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="detail-card">
                    <p>Vocabulary Feedback</p>
                    <ul>
                      {(scores.detailed_feedback.vocabulary_feedback || []).map((item, idx) => (
                        <li key={`vocab-${idx}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="detail-card">
                    <p>Grammar Feedback</p>
                    <ul>
                      {(scores.detailed_feedback.grammar_feedback || []).map((item, idx) => (
                        <li key={`grammar-${idx}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="detail-card highlight">
                    <p>Improvement Tips</p>
                    <ul>
                      {(scores.detailed_feedback.improvement_tips || []).map((item, idx) => (
                        <li key={`tip-${idx}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {scores.detailed_feedback?.rewrite_examples && (
                <div className="rewrite-panel">
                  <p>Instead of Saying X, Say Y</p>
                  <div className="rewrite-list">
                    {(scores.detailed_feedback.rewrite_examples || []).map((item, idx) => (
                      <div key={`rewrite-${idx}`} className="rewrite-item">
                        <strong>Instead of:</strong>
                        <span>{item.from}</span>
                        <strong>Say:</strong>
                        <span>{item.to}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {scores.detailed_feedback?.filler_highlights && (
                <div className="filler-highlight-panel">
                  <p>Repeated Filler Words From Transcript</p>
                  <div className="filler-chip-row">
                    {(scores.detailed_feedback.filler_highlights || []).map((item) => (
                      <span key={item.word} className="filler-chip">
                        {item.word} ({item.count}x)
                      </span>
                    ))}
                  </div>
                  {scores.detailed_feedback.highlighted_transcript && (
                    <pre className="highlighted-transcript">
                      {scores.detailed_feedback.highlighted_transcript}
                    </pre>
                  )}
                </div>
              )}
            </div>
          )}

          <div className="history-journal card-lite">
            <div className="history-journal-head">
              <h4>Progress Journal</h4>
              <span>{historyLoading ? 'Refreshing...' : `${savedSessions.length} sessions`}</span>
            </div>

            <div className="journal-filters" role="group" aria-label="Journal filters">
              <label>
                Range
                <select value={journalRange} onChange={(e) => setJournalRange(e.target.value)}>
                  <option value="7d">Last 7 days</option>
                  <option value="all">All time</option>
                </select>
              </label>
              <label>
                Mode
                <select value={journalMode} onChange={(e) => setJournalMode(e.target.value)}>
                  <option value="all">All sessions</option>
                  <option value="live">Live only</option>
                  <option value="upload">Upload only</option>
                </select>
              </label>
              <label>
                Sort
                <select value={journalSort} onChange={(e) => setJournalSort(e.target.value)}>
                  <option value="latest">Latest first</option>
                  <option value="best">Best overall</option>
                </select>
              </label>
            </div>

            {overallDelta !== null && (
              <p className={`delta-badge ${overallDelta >= 0 ? 'up' : 'down'}`}>
                Latest overall {overallDelta >= 0 ? '+' : ''}{overallDelta.toFixed(2)} vs previous session
              </p>
            )}

            <div className="compare-toolbar">
              <div>
                <strong>Compare Sessions</strong>
                <p>Select up to 3 sessions to compare side by side.</p>
              </div>
              <div className="compare-toolbar-actions">
                <button className="mini-btn" onClick={() => exportCompareSessions('json')} disabled={compareSessions.length === 0}>
                  Export JSON
                </button>
                <button className="mini-btn" onClick={() => exportCompareSessions('csv')} disabled={compareSessions.length === 0}>
                  Export CSV
                </button>
                <button className="mini-btn" onClick={() => setCompareSessionIds([])} disabled={compareSessionIds.length === 0}>
                  Clear Compare
                </button>
              </div>
            </div>

            {compareSessions.length >= 2 && (
              <div className="compare-panel">
                <div className="compare-panel-head">
                  <strong>Comparison View</strong>
                  <span>{compareSessions.length} selected</span>
                </div>

                {compareSummary && (
                  <div className="compare-summary-grid">
                    <div><span>Avg Fluency</span><strong>{compareSummary.fluency.toFixed(1)}</strong></div>
                    <div><span>Avg Vocabulary</span><strong>{compareSummary.vocabulary.toFixed(1)}</strong></div>
                    <div><span>Avg Communication</span><strong>{compareSummary.communication.toFixed(1)}</strong></div>
                    <div><span>Avg Overall</span><strong>{compareSummary.overall.toFixed(1)}</strong></div>
                  </div>
                )}

                <div className="compare-cards">
                  {compareSessions.map((item) => (
                    <div key={item.session_id} className="compare-card">
                      <div className="compare-card-head">
                        <strong>{item.mode || 'session'}</strong>
                        <button className="mini-btn" onClick={() => toggleCompareSession(item.session_id)}>Remove</button>
                      </div>
                      <p>{item.created_at ? new Date(item.created_at).toLocaleString() : 'Unknown date'}</p>
                      <div className="compare-score-row">
                        <span>Overall</span>
                        <strong>{Number(item?.scores?.overall || 0).toFixed(1)}</strong>
                      </div>
                      <div className="compare-score-row">
                        <span>Fluency</span>
                        <strong>{Number(item?.scores?.fluency || 0).toFixed(1)}</strong>
                      </div>
                      <div className="compare-score-row">
                        <span>Vocabulary</span>
                        <strong>{Number(item?.scores?.vocabulary || 0).toFixed(1)}</strong>
                      </div>
                      <div className="compare-score-row">
                        <span>Communication</span>
                        <strong>{Number(item?.scores?.communication || 0).toFixed(1)}</strong>
                      </div>
                      <TinySparkline values={item.overall_history || []} width={150} height={36} />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {groupedSessions.length === 0 ? (
              <span className="placeholder">Complete a live or upload session to build history.</span>
            ) : (
              <div className="journal-groups">
                {groupedSessions.map((group) => (
                  <div key={group.date} className="journal-group">
                    <div className="journal-group-head">
                      <strong>{group.date}</strong>
                      <span>Avg overall {Number(group.avgOverall).toFixed(1)}</span>
                    </div>
                    <div className="journal-items">
                      {group.items.slice(0, 4).map((item) => (
                        <div key={item.session_id} className="journal-item">
                          <div className="journal-item-main">
                            <span className="mode-tag">{item.mode || 'session'}</span>
                            <span>Overall {Number(item?.scores?.overall || 0).toFixed(1)}</span>
                            <span>WPM {Number(item?.metrics?.wpm || 0).toFixed(0)}</span>
                            <label className="compare-check">
                              <input
                                type="checkbox"
                                checked={compareSessionIds.includes(item.session_id)}
                                onChange={() => toggleCompareSession(item.session_id)}
                              />
                              Compare
                            </label>
                          </div>
                          <TinySparkline values={item.overall_history || []} width={120} height={28} />
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {sessionId && <p className="session-id">Session: {sessionId}</p>}
          {error && <p className="error-text">{error}</p>}
        </section>

        <section className="right-column">
          <TranscriptPanel
            title="Original Transcript"
            text={originalTranscript}
            variant="original"
          />
          <TranscriptPanel
            title="Refined Transcript"
            text={refinedTranscript}
            variant="refined"
          />
          <SummaryPanel
            summary={summaryText}
            onSummarize={handleSummarizeTranscript}
            isSummarizing={isSummarizing}
            sourceWordCount={summarySourceWordCount}
            summaryWordCount={summaryWordCount}
            error={summaryError}
            disabled={!refinedTranscript.trim()}
          />
        </section>
      </main>
    </div>
  );
}

export default App;
