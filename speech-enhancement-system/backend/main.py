"""FastAPI backend for realtime/file denoise + transcription using DeepFilterNet2."""

from __future__ import annotations

import json
import os
import re
import uuid
import logging
import shutil
from threading import Lock
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from denoise_file_df2 import DF2Denoiser, denoise_file, _resample_if_needed
from scoring import compute_scores


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SESSION_HISTORY_PATH = OUTPUT_DIR / "session_history.json"

DEEPFILTER_MODEL_BASE_DIR = str(Path(__file__).resolve().parent / "models")
DEEPFILTER_DEVICE = "auto"
DEEPFILTER_EPOCH = "best"

WHISPER_MODEL = "tiny"
WHISPER_DEVICE = "cpu"

SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
SUMMARIZER_MIN_WORDS = 40
SUMMARIZER_MAX_CHUNK_TOKENS = 900

history_lock = Lock()


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1)


class TextSummarizer:
    def __init__(self, model_name: str = SUMMARIZER_MODEL, device: str = "cpu"):
        logger.info("Loading summarizer model %s on %s...", model_name, device)
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipeline_device = 0 if device != "cpu" else -1
        self.pipeline = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=pipeline_device,
        )
        logger.info("Loaded summarizer model %s successfully", model_name)

    def _chunk_text(self, text: str) -> list[str]:
        token_ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        if not token_ids:
            return []

        chunks: list[str] = []
        for start in range(0, len(token_ids), SUMMARIZER_MAX_CHUNK_TOKENS):
            chunk_ids = token_ids[start : start + SUMMARIZER_MAX_CHUNK_TOKENS]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
            if chunk_text:
                chunks.append(chunk_text)
        return chunks

    def summarize(self, text: str) -> str:
        normalized_text = " ".join(str(text or "").split())
        if not normalized_text:
            return ""

        if len(normalized_text.split()) < SUMMARIZER_MIN_WORDS:
            return normalized_text

        chunk_summaries: list[str] = []
        for chunk in self._chunk_text(normalized_text):
            result = self.pipeline(
                chunk,
                max_length=130,
                min_length=30,
                do_sample=False,
                truncation=True,
            )
            summary_text = result[0]["summary_text"].strip()
            if summary_text:
                chunk_summaries.append(summary_text)

        if not chunk_summaries:
            return normalized_text

        combined_summary = " ".join(chunk_summaries).strip()
        if len(chunk_summaries) > 1 and len(combined_summary.split()) >= SUMMARIZER_MIN_WORDS:
            result = self.pipeline(
                combined_summary,
                max_length=130,
                min_length=30,
                do_sample=False,
                truncation=True,
            )
            combined_summary = result[0]["summary_text"].strip()

        return " ".join(combined_summary.split())


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "audio_chunks": [],
            "cleaned_audio_chunks": [],
            "sample_rate": SAMPLE_RATE,
            "total_cleaned_samples": 0,
            "original_transcript": [],
            "refined_transcript": [],
            "analysis_windows": 0,
            "silent_windows": 0,
            "score_history": [],
            "custom_filters": {},
            "is_active": True,
        }
        return session_id

    def get(self, session_id: str) -> Optional[Dict]:
        return self.sessions.get(session_id)

    def close(self, session_id: str) -> None:
        if session_id in self.sessions:
            self.sessions[session_id]["is_active"] = False


def _load_session_history() -> List[Dict]:
    if not SESSION_HISTORY_PATH.exists():
        return []
    try:
        payload = json.loads(SESSION_HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        return []
    except Exception:
        return []


def _save_session_history(entries: List[Dict]) -> None:
    SESSION_HISTORY_PATH.write_text(json.dumps(entries, ensure_ascii=True, indent=2), encoding="utf-8")


def _persist_session_result(
    session_id: str,
    mode: str,
    original_text: str,
    refined_text: str,
    scores: Dict,
    score_history: List[Dict],
) -> None:
    record = {
        "session_id": session_id,
        "mode": mode,
        "created_at": datetime.now().isoformat(),
        "scores": {
            "fluency": scores.get("fluency", 0.0),
            "vocabulary": scores.get("vocabulary", 0.0),
            "communication": scores.get("communication", 0.0),
            "overall": scores.get("overall", 0.0),
            "communication_level": scores.get("communication_level", "A1"),
        },
        "metrics": scores.get("metrics", {}),
        "feedback": scores.get("feedback", []),
        "overall_feedback": scores.get("overall_feedback", ""),
        "detailed_feedback": scores.get("detailed_feedback", {}),
        "original_preview": (original_text or "")[:280],
        "refined_preview": (refined_text or "")[:280],
        "overall_history": [float(p.get("overall", 0.0)) for p in (score_history or [])],
        "history_points": len(score_history or []),
    }

    with history_lock:
        history = _load_session_history()
        history.append(record)
        # Keep newest 500 records to avoid unbounded growth.
        if len(history) > 500:
            history = history[-500:]
        _save_session_history(history)


def _normalize_custom_filters(filters: Optional[Dict[str, Optional[str]]]) -> Dict[str, Optional[str]]:
    normalized: Dict[str, Optional[str]] = {}
    for word, replacement in (filters or {}).items():
        clean_word = str(word or "").strip()
        if not clean_word:
            continue
        clean_replacement = "" if replacement is None else str(replacement).strip()
        normalized[clean_word] = clean_replacement if clean_replacement else None
    return normalized


def _apply_custom_filters(text: str, filters: Optional[Dict[str, Optional[str]]]) -> str:
    if not text:
        return ""

    result = str(text)
    for word, replacement in _normalize_custom_filters(filters).items():
        pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
        result = pattern.sub(replacement or "", result)

    result = " ".join(result.split())
    if result:
        result = result[0].upper() + result[1:]
    return result


def _get_session_filters(session: Optional[Dict]) -> Dict[str, Optional[str]]:
    if not session:
        return {}
    return _normalize_custom_filters(session.get("custom_filters", {}))


def _rebuild_refined_transcript(session: Dict) -> str:
    filters = _get_session_filters(session)
    original_chunks = session.get("original_transcript", [])
    refined_chunks = [_apply_custom_filters(chunk, filters) for chunk in original_chunks if chunk]
    session["refined_transcript"] = [chunk for chunk in refined_chunks if chunk]
    return " ".join(session["refined_transcript"]).strip()


def _parse_custom_filters(raw_filters: Optional[str]) -> Dict[str, Optional[str]]:
    if not raw_filters:
        return {}

    try:
        parsed = json.loads(raw_filters)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="custom_filters must be valid JSON") from exc

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="custom_filters must be a JSON object")

    return _normalize_custom_filters(parsed)


class TextProcessor:
    def __init__(self):
        self.custom_filters: Dict[str, Optional[str]] = {}

    def set_custom_filters(self, filters: Dict[str, Optional[str]]) -> None:
        self.custom_filters = filters or {}

    def process(self, text: str) -> str:
        if not text:
            return ""

        result = text
        for word, replacement in self.custom_filters.items():
            if not word:
                continue
            if replacement is None:
                replacement = ""
            result = result.replace(word, replacement)
            result = result.replace(word.lower(), replacement)
            result = result.replace(word.upper(), replacement)

        result = " ".join(result.split())
        if result:
            result = result[0].upper() + result[1:]
        return result


class Transcriber:
    def __init__(self, model_size: str = "tiny", device: str = "cpu"):
        logger.info("Loading transcriber model %s on %s...", model_size, device)
        from faster_whisper import WhisperModel

        compute_type = "int8" if device == "cpu" else "float16"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Loaded faster-whisper %s successfully", model_size)

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return ""

        if sample_rate != SAMPLE_RATE:
            audio = _resample_if_needed(audio, sample_rate, SAMPLE_RATE)

        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        text = " ".join((seg.text or "").strip() for seg in segments).strip()
        return " ".join(text.split())


class StreamingTranscriber:
    def __init__(self, transcriber: Transcriber, sample_rate: int = SAMPLE_RATE, window_s: float = 1.2):
        self.transcriber = transcriber
        self.sample_rate = sample_rate
        self.window_samples = int(window_s * sample_rate)
        self.buffer = np.array([], dtype=np.float32)

    def add_chunk(self, chunk: np.ndarray) -> Optional[str]:
        chunk = np.asarray(chunk, dtype=np.float32)
        self.buffer = np.concatenate([self.buffer, chunk])
        if len(self.buffer) < self.window_samples:
            return None

        audio = self.buffer[: self.window_samples]
        self.buffer = self.buffer[self.window_samples :]
        return self.transcriber.transcribe(audio, self.sample_rate)

    def flush(self) -> Optional[str]:
        if len(self.buffer) == 0:
            return None
        audio = self.buffer.copy()
        self.buffer = np.array([], dtype=np.float32)
        return self.transcriber.transcribe(audio, self.sample_rate)

    def reset(self) -> None:
        self.buffer = np.array([], dtype=np.float32)


app = FastAPI(title="Speech Enhancement API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_manager = SessionManager()
shared_denoiser: Optional[DF2Denoiser] = None
shared_transcriber: Optional[Transcriber] = None
streaming_transcriber: Optional[StreamingTranscriber] = None
text_processor = TextProcessor()
shared_summarizer: Optional[TextSummarizer] = None
summarizer_lock = Lock()


def _get_shared_summarizer() -> TextSummarizer:
    global shared_summarizer
    if shared_summarizer is None:
        with summarizer_lock:
            if shared_summarizer is None:
                shared_summarizer = TextSummarizer(device="cpu")
    return shared_summarizer


def _append_score_history(session: Dict, score_payload: Dict, timestamp: str) -> Dict:
    history = session.setdefault("score_history", [])
    point = {
        "step": len(history) + 1,
        "timestamp": timestamp,
        "fluency": score_payload.get("fluency", 0.0),
        "vocabulary": score_payload.get("vocabulary", 0.0),
        "communication": score_payload.get("communication", 0.0),
        "overall": score_payload.get("overall", 0.0),
    }
    history.append(point)
    return point


def _save_transcript_file(session_id: str, original: str, refined: str, scores: Optional[Dict] = None) -> Path:
    path = OUTPUT_DIR / f"refined_transcript_{session_id}.txt"
    score_block = ""
    if scores:
        detailed = scores.get("detailed_feedback", {}) if isinstance(scores.get("detailed_feedback", {}), dict) else {}
        pace_feedback = detailed.get("pace_feedback", []) or []
        vocabulary_feedback = detailed.get("vocabulary_feedback", []) or []
        grammar_feedback = detailed.get("grammar_feedback", []) or []
        rewrite_examples = detailed.get("rewrite_examples", []) or []
        filler_highlights = detailed.get("filler_highlights", []) or []
        highlighted_transcript = detailed.get("highlighted_transcript", "") or ""
        improvement_tips = detailed.get("improvement_tips", []) or []
        overall_feedback = scores.get("overall_feedback", "")

        score_block = (
            "COMMUNICATION SCORES\n"
            + "-" * 60
            + "\n"
            + f"Fluency Score: {scores.get('fluency', 0):.2f}/100\n"
            + f"Vocabulary Score: {scores.get('vocabulary', 0):.2f}/100\n"
            + f"Communication Score: {scores.get('communication', 0):.2f}/100\n"
            + f"Overall Score: {scores.get('overall', 0):.2f}/100\n\n"
            + f"Communication Level: {scores.get('communication_level', 'A1')} - {scores.get('communication_level_label', '')}\n"
            + f"Level Description: {scores.get('communication_level_description', '')}\n\n"
            + f"Overall Feedback: {overall_feedback}\n\n"
            + "Detailed Feedback\n"
            + "-" * 60
            + "\n"
            + "Pace Feedback\n"
            + "\n".join(f"- {item}" for item in pace_feedback)
            + "\n\nVocabulary Feedback\n"
            + "\n".join(f"- {item}" for item in vocabulary_feedback)
            + "\n\nGrammar Feedback\n"
            + "\n".join(f"- {item}" for item in grammar_feedback)
            + "\n\nInstead Of Saying X, Say Y\n"
            + "\n".join(f"- {item.get('from', '')} -> {item.get('to', '')}" for item in rewrite_examples)
            + "\n\nFiller Highlights\n"
            + "\n".join(f"- {item.get('word', '')} ({item.get('count', 0)}x)" for item in filler_highlights)
            + "\n\nHighlighted Transcript\n"
            + highlighted_transcript
            + "\n\nImprovement Tips\n"
            + "\n".join(f"- {item}" for item in improvement_tips)
            + "\n\n"
            + "Insights:\n"
            + "\n".join(f"- {item}" for item in scores.get("feedback", []))
            + "\n\n"
        )
    content = (
        "SPEECH ENHANCEMENT TRANSCRIPT\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        + score_block
        +
        "ORIGINAL TRANSCRIPT\n"
        "-" * 60
        + "\n"
        + original
        + "\n\n"
        + "REFINED TRANSCRIPT\n"
        + "-" * 60
        + "\n"
        + refined
        + "\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


@app.on_event("startup")
async def startup_event():
    global shared_denoiser, shared_transcriber, streaming_transcriber
    logger.info("Initializing speech enhancement system...")
    shared_denoiser = DF2Denoiser(
        model_dir=DEEPFILTER_MODEL_BASE_DIR,
        device=DEEPFILTER_DEVICE,
        epoch=DEEPFILTER_EPOCH,
    )
    shared_transcriber = Transcriber(model_size=WHISPER_MODEL, device=WHISPER_DEVICE)
    streaming_transcriber = StreamingTranscriber(shared_transcriber, sample_rate=SAMPLE_RATE)
    logger.info("System initialized successfully")


@app.get("/")
async def root():
    return {"status": "healthy", "service": "Speech Enhancement API", "version": "2.0.0"}


@app.get("/health")
async def health():
    active = len([s for s in session_manager.sessions.values() if s.get("is_active")])
    return {
        "status": "healthy",
        "models_loaded": {
            "denoiser": shared_denoiser is not None,
            "transcriber": shared_transcriber is not None,
            "summarizer": shared_summarizer is not None,
        },
        "active_sessions": active,
    }


@app.post("/summarize")
async def summarize_transcript(payload: SummarizeRequest):
    text = " ".join(payload.text.split())
    if not text:
        raise HTTPException(status_code=400, detail="No transcript text provided")

    try:
        summarizer = _get_shared_summarizer()
        summary = summarizer.summarize(text)
    except Exception as exc:
        logger.error("Summarization failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "summary": summary,
        "model": SUMMARIZER_MODEL,
        "source_word_count": len(text.split()),
        "summary_word_count": len(summary.split()) if summary else 0,
    }


@app.get("/sessions/history")
async def sessions_history(limit: int = 50):
    limit = max(1, min(int(limit), 200))
    with history_lock:
        history = _load_session_history()
    history = sorted(history, key=lambda x: x.get("created_at", ""), reverse=True)
    return {
        "count": len(history),
        "items": history[:limit],
    }


@app.get("/sessions/history/{session_id}")
async def session_history_item(session_id: str):
    with history_lock:
        history = _load_session_history()
    match = next((item for item in history if item.get("session_id") == session_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Session history not found")
    return match


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    session_id = session_manager.create_session()
    session = session_manager.get(session_id)

    if streaming_transcriber is not None:
        streaming_transcriber.reset()

    await websocket.send_json({"type": "session_started", "session_id": session_id})

    try:
        while True:
            message = await websocket.receive_json()
            m_type = message.get("type")

            if m_type == "config":
                filters = _normalize_custom_filters(message.get("custom_filters", {}))
                session["custom_filters"] = filters
                text_processor.set_custom_filters(filters)
                refined_transcript = _rebuild_refined_transcript(session)
                await websocket.send_json(
                    {
                        "type": "config_updated",
                        "session_id": session_id,
                        "refined_transcript": refined_transcript,
                    }
                )
                continue

            if m_type == "audio_chunk":
                if shared_denoiser is None or streaming_transcriber is None:
                    raise RuntimeError("Models are not initialized")

                audio_data = np.array(message.get("data", []), dtype=np.float32)
                sample_rate = int(message.get("sample_rate", SAMPLE_RATE))
                session["sample_rate"] = sample_rate
                if audio_data.size == 0:
                    continue

                cleaned_audio = shared_denoiser.denoise_chunk(audio_data, sample_rate)
                session["audio_chunks"].append(audio_data.tolist())
                session["cleaned_audio_chunks"].append(cleaned_audio.tolist())
                session["total_cleaned_samples"] = int(session.get("total_cleaned_samples", 0)) + int(len(cleaned_audio))

                text = streaming_transcriber.add_chunk(cleaned_audio)
                if text is not None:
                    session["analysis_windows"] = int(session.get("analysis_windows", 0)) + 1
                original_text = (text or "").strip()
                refined_text = ""
                if original_text:
                    refined_text = _apply_custom_filters(original_text, session.get("custom_filters", {}))
                    session["original_transcript"].append(original_text)
                    session["refined_transcript"].append(refined_text)
                elif text is not None:
                    session["silent_windows"] = int(session.get("silent_windows", 0)) + 1

                rolling_score = None
                rolling_history_point = None
                if text is not None:
                    running_original = " ".join(session.get("original_transcript", []))
                    running_refined = " ".join(session.get("refined_transcript", []))
                    running_duration = float(session.get("total_cleaned_samples", 0)) / float(SAMPLE_RATE)
                    rolling_score = compute_scores(
                        text=running_original,
                        cleaned_text=running_refined,
                        duration_sec=running_duration,
                        pause_count=int(session.get("silent_windows", 0)),
                    )
                    rolling_history_point = _append_score_history(session, rolling_score, datetime.now().isoformat())

                await websocket.send_json(
                    {
                        "type": "processed",
                        "session_id": session_id,
                        "cleaned_audio": cleaned_audio.tolist(),
                        "original_text": original_text,
                        "refined_text": refined_text,
                        "rolling_scores": rolling_score,
                        "score_history_point": rolling_history_point,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                if rolling_score is not None:
                    await websocket.send_json(
                        {
                            "type": "score_update",
                            "session_id": session_id,
                            "scores": rolling_score,
                            "history_point": rolling_history_point,
                        }
                    )
                continue

            if m_type == "end_session":
                final_text = ""
                if streaming_transcriber is not None:
                    final_text = (streaming_transcriber.flush() or "").strip()
                if final_text:
                    session["original_transcript"].append(final_text)
                    session["refined_transcript"].append(
                        _apply_custom_filters(final_text, session.get("custom_filters", {}))
                    )
                elif streaming_transcriber is not None:
                    session["silent_windows"] = int(session.get("silent_windows", 0)) + 1

                cleaned_chunks = session.get("cleaned_audio_chunks", [])
                cleaned_audio = (
                    np.concatenate([np.asarray(c, dtype=np.float32) for c in cleaned_chunks])
                    if cleaned_chunks
                    else np.array([], dtype=np.float32)
                )
                original_chunks = session.get("audio_chunks", [])
                original_audio = (
                    np.concatenate([np.asarray(c, dtype=np.float32) for c in original_chunks])
                    if original_chunks
                    else np.array([], dtype=np.float32)
                )
                stream_sample_rate = int(session.get("sample_rate", SAMPLE_RATE))
                cleaned_audio_path = OUTPUT_DIR / f"cleaned_audio_{session_id}.wav"
                original_audio_path = OUTPUT_DIR / f"original_audio_{session_id}.wav"
                sf.write(str(cleaned_audio_path), cleaned_audio, stream_sample_rate)
                sf.write(str(original_audio_path), original_audio, stream_sample_rate)

                original = " ".join(session.get("original_transcript", []))
                refined = " ".join(session.get("refined_transcript", []))
                duration_sec = float(len(cleaned_audio) / SAMPLE_RATE) if cleaned_audio.size else 0.0
                score_payload = compute_scores(
                    text=original,
                    cleaned_text=refined,
                    duration_sec=duration_sec,
                    pause_count=int(session.get("silent_windows", 0)),
                )
                final_point = _append_score_history(session, score_payload, datetime.now().isoformat())

                _save_transcript_file(session_id, original, refined, score_payload)
                _persist_session_result(
                    session_id=session_id,
                    mode="live",
                    original_text=original,
                    refined_text=refined,
                    scores=score_payload,
                    score_history=session.get("score_history", []),
                )

                await websocket.send_json(
                    {
                        "type": "session_ended",
                        "session_id": session_id,
                        "scores": score_payload,
                        "score_history": session.get("score_history", []),
                        "final_history_point": final_point,
                        "download_urls": {
                            "audio": f"/download/audio/{session_id}",
                            "original_audio": f"/download/audio/original/{session_id}",
                            "cleaned_audio": f"/download/audio/cleaned/{session_id}",
                            "transcript": f"/download/transcript/{session_id}",
                        },
                    }
                )
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session %s", session_id)
    except Exception as exc:
        logger.error("WebSocket error for session %s: %s", session_id, exc)
        await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        session_manager.close(session_id)


@app.post("/upload")
async def upload_audio_file(file: UploadFile = File(...), custom_filters: str = Form(default="")):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".wav", ".mp3", ".flac", ".m4a"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    if shared_transcriber is None:
        raise HTTPException(status_code=500, detail="Transcriber not initialized")

    session_id = session_manager.create_session()
    temp_path = OUTPUT_DIR / f"temp_{session_id}{ext}"
    cleaned_audio_path = OUTPUT_DIR / f"cleaned_audio_{session_id}.wav"
    original_audio_path = OUTPUT_DIR / f"original_audio_{session_id}.wav"
    filters = _parse_custom_filters(custom_filters)

    try:
        with open(temp_path, "wb") as fh:
            fh.write(await file.read())

        denoise_file(
            input_path=temp_path,
            output_path=cleaned_audio_path,
            model_dir=DEEPFILTER_MODEL_BASE_DIR,
            device=DEEPFILTER_DEVICE,
            epoch=DEEPFILTER_EPOCH,
        )

        try:
            original_audio, original_sr = sf.read(str(temp_path), dtype="float32")
            if original_audio.ndim > 1:
                original_audio = np.mean(original_audio, axis=1)
            sf.write(str(original_audio_path), original_audio, original_sr)
        except Exception:
            shutil.copy2(temp_path, original_audio_path)

        cleaned_audio, cleaned_sr = sf.read(str(cleaned_audio_path), dtype="float32")
        if cleaned_audio.ndim > 1:
            cleaned_audio = np.mean(cleaned_audio, axis=1)

        original_text = shared_transcriber.transcribe(cleaned_audio, cleaned_sr)
        refined_text = _apply_custom_filters(original_text, filters)

        duration_sec = float(len(cleaned_audio) / cleaned_sr) if cleaned_sr > 0 else 0.0
        score_payload = compute_scores(
            text=original_text,
            cleaned_text=refined_text,
            duration_sec=duration_sec,
            pause_count=0,
        )
        score_history = [
            {
                "step": 1,
                "timestamp": datetime.now().isoformat(),
                "fluency": score_payload.get("fluency", 0.0),
                "vocabulary": score_payload.get("vocabulary", 0.0),
                "communication": score_payload.get("communication", 0.0),
                "overall": score_payload.get("overall", 0.0),
            }
        ]

        _save_transcript_file(session_id, original_text, refined_text, score_payload)
        _persist_session_result(
            session_id=session_id,
            mode="upload",
            original_text=original_text,
            refined_text=refined_text,
            scores=score_payload,
            score_history=score_history,
        )

        return {
            "session_id": session_id,
            "original_transcript": original_text,
            "refined_transcript": refined_text,
            "scores": score_payload,
            "score_history": score_history,
            "download_urls": {
                "audio": f"/download/audio/{session_id}",
                "original_audio": f"/download/audio/original/{session_id}",
                "cleaned_audio": f"/download/audio/cleaned/{session_id}",
                "transcript": f"/download/transcript/{session_id}",
            },
        }

    except Exception as exc:
        logger.error("Upload processing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.get("/download/audio/{session_id}")
async def download_audio(session_id: str):
    path = OUTPUT_DIR / f"cleaned_audio_{session_id}.wav"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(path), media_type="audio/wav", filename=path.name)


@app.get("/download/audio/original/{session_id}")
async def download_original_audio(session_id: str):
    path = OUTPUT_DIR / f"original_audio_{session_id}.wav"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Original audio file not found")
    return FileResponse(str(path), media_type="audio/wav", filename=path.name)


@app.get("/download/audio/cleaned/{session_id}")
async def download_cleaned_audio(session_id: str):
    path = OUTPUT_DIR / f"cleaned_audio_{session_id}.wav"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Cleaned audio file not found")
    return FileResponse(str(path), media_type="audio/wav", filename=path.name)


@app.get("/download/transcript/{session_id}")
async def download_transcript(session_id: str):
    path = OUTPUT_DIR / f"refined_transcript_{session_id}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found")
    return FileResponse(str(path), media_type="text/plain", filename=path.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
