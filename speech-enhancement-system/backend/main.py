"""FastAPI backend for realtime/file denoise + transcription using DeepFilterNet2."""

from __future__ import annotations

import uuid
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from denoise_file_df2 import DF2Denoiser, denoise_file, _resample_if_needed


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEEPFILTER_MODEL_BASE_DIR = str(Path(__file__).resolve().parent / "models")
DEEPFILTER_DEVICE = "auto"
DEEPFILTER_EPOCH = "best"

WHISPER_MODEL = "tiny"
WHISPER_DEVICE = "cpu"


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
            "original_transcript": [],
            "refined_transcript": [],
            "custom_filters": {},
            "is_active": True,
        }
        return session_id

    def get(self, session_id: str) -> Optional[Dict]:
        return self.sessions.get(session_id)

    def close(self, session_id: str) -> None:
        if session_id in self.sessions:
            self.sessions[session_id]["is_active"] = False


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


class PassthroughDenoiser:
    """Fallback denoiser used when DeepFilterNet weights are unavailable."""

    def denoise_chunk(self, audio_chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.asarray(audio_chunk, dtype=np.float32)

    def denoise_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.asarray(audio, dtype=np.float32)


def _build_denoiser() -> object:
    try:
        return DF2Denoiser(
            model_dir=DEEPFILTER_MODEL_BASE_DIR,
            device=DEEPFILTER_DEVICE,
            epoch=DEEPFILTER_EPOCH,
        )
    except FileNotFoundError as exc:
        logger.warning("DeepFilterNet model not found, falling back to passthrough denoiser: %s", exc)
    except Exception as exc:
        logger.warning("DeepFilterNet failed to initialize, falling back to passthrough denoiser: %s", exc)

    return PassthroughDenoiser()


def _denoise_uploaded_file(input_path: Path, output_path: Path) -> None:
    if isinstance(shared_denoiser, DF2Denoiser):
        denoise_file(
            input_path=input_path,
            output_path=output_path,
            model_dir=DEEPFILTER_MODEL_BASE_DIR,
            device=DEEPFILTER_DEVICE,
            epoch=DEEPFILTER_EPOCH,
        )
        return

    shutil.copy2(input_path, output_path)


def _save_transcript_file(session_id: str, original: str, refined: str) -> Path:
    path = OUTPUT_DIR / f"refined_transcript_{session_id}.txt"
    content = (
        "SPEECH ENHANCEMENT TRANSCRIPT\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
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
    shared_denoiser = _build_denoiser()
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
        },
        "active_sessions": active,
    }


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
                filters = message.get("custom_filters", {})
                session["custom_filters"] = filters
                text_processor.set_custom_filters(filters)
                await websocket.send_json({"type": "config_updated", "session_id": session_id})
                continue

            if m_type == "audio_chunk":
                if shared_denoiser is None or streaming_transcriber is None:
                    raise RuntimeError("Models are not initialized")

                audio_data = np.array(message.get("data", []), dtype=np.float32)
                sample_rate = int(message.get("sample_rate", SAMPLE_RATE))
                if audio_data.size == 0:
                    continue

                cleaned_audio = shared_denoiser.denoise_chunk(audio_data, sample_rate)
                session["audio_chunks"].append(audio_data.tolist())
                session["cleaned_audio_chunks"].append(cleaned_audio.tolist())

                text = streaming_transcriber.add_chunk(cleaned_audio)
                original_text = (text or "").strip()
                refined_text = ""
                if original_text:
                    refined_text = text_processor.process(original_text)
                    session["original_transcript"].append(original_text)
                    session["refined_transcript"].append(refined_text)

                await websocket.send_json(
                    {
                        "type": "processed",
                        "session_id": session_id,
                        "cleaned_audio": cleaned_audio.tolist(),
                        "original_text": original_text,
                        "refined_text": refined_text,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                continue

            if m_type == "end_session":
                final_text = ""
                if streaming_transcriber is not None:
                    final_text = (streaming_transcriber.flush() or "").strip()
                if final_text:
                    session["original_transcript"].append(final_text)
                    session["refined_transcript"].append(text_processor.process(final_text))

                cleaned_chunks = session.get("cleaned_audio_chunks", [])
                cleaned_audio = (
                    np.concatenate([np.asarray(c, dtype=np.float32) for c in cleaned_chunks])
                    if cleaned_chunks
                    else np.array([], dtype=np.float32)
                )
                audio_path = OUTPUT_DIR / f"cleaned_audio_{session_id}.wav"
                sf.write(str(audio_path), cleaned_audio, SAMPLE_RATE)

                original = " ".join(session.get("original_transcript", []))
                refined = " ".join(session.get("refined_transcript", []))
                _save_transcript_file(session_id, original, refined)

                await websocket.send_json(
                    {
                        "type": "session_ended",
                        "session_id": session_id,
                        "download_urls": {
                            "audio": f"/download/audio/{session_id}",
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
async def upload_audio_file(file: UploadFile = File(...)):
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

    try:
        with open(temp_path, "wb") as fh:
            fh.write(await file.read())

        _denoise_uploaded_file(temp_path, cleaned_audio_path)

        cleaned_audio, cleaned_sr = sf.read(str(cleaned_audio_path), dtype="float32")
        if cleaned_audio.ndim > 1:
            cleaned_audio = np.mean(cleaned_audio, axis=1)

        original_text = shared_transcriber.transcribe(cleaned_audio, cleaned_sr)
        refined_text = text_processor.process(original_text)

        _save_transcript_file(session_id, original_text, refined_text)

        return {
            "session_id": session_id,
            "original_transcript": original_text,
            "refined_transcript": refined_text,
            "download_urls": {
                "audio": f"/download/audio/{session_id}",
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


@app.get("/download/transcript/{session_id}")
async def download_transcript(session_id: str):
    path = OUTPUT_DIR / f"refined_transcript_{session_id}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found")
    return FileResponse(str(path), media_type="text/plain", filename=path.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
