"""video_transcribe : les paroles d'une vidéo (spécification, section 3.3 ; lot 4b).

Faster-Whisper, modèle small en int8 sur processeur, chargé une seule fois par
travailleur. La langue est détectée ; les paroles sont rendues par segments
horodatés, et Howé y applique ses règles (un numéro dicté coupé en deux segments
compris). Le texte n'est écrit dans aucun journal.

Le modèle n'est jamais téléchargé pendant une tâche : il doit être sur le serveur
(ou dans le cache), sauf WHISPER_DOWNLOAD=1.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .client import DownloadRefused, HoweClient
from .config import Settings
from .logs import step
from .photo_check import Cancelled, TaskFailed

_model = None
_model_lock = threading.Lock()


def model(settings: Settings):
    """Le modèle Whisper, chargé à la première transcription, puis gardé."""
    global _model
    with _model_lock:
        if _model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise TaskFailed("asr_unavailable", True, "faster-whisper absent (requirements-asr.txt)") from exc
            try:
                # Processeur et int8 par défaut ; sur RunPod (lot 6), GPU et float16.
                _model = WhisperModel(settings.whisper_model, device=settings.whisper_device, compute_type=settings.whisper_compute_type,
                                      cpu_threads=settings.whisper_threads, local_files_only=settings.whisper_local_only)
            except Exception as exc:  # noqa: BLE001 — modèle absent ou abîmé : panne de configuration
                raise TaskFailed("asr_model_missing", True, type(exc).__name__) from exc
        return _model


def read_wav(path: Path):
    """Le son préparé par video_prepare (WAV PCM 16 bits, mono, 16 kHz), en échantillons pour Whisper.

    Lu ici plutôt que par PyAV : faster-whisper 1.1.1 appelle une option que les versions
    récentes de PyAV ont retirée.
    """
    import wave

    import numpy as np

    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getsampwidth() != 2 or wav.getnchannels() != 1 or wav.getframerate() != 16000:
                raise TaskFailed("audio_format", False, "WAV 16 bits mono 16 kHz attendu")
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise TaskFailed("audio_unreadable", True, type(exc).__name__) from exc
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def run(task_input: dict[str, Any], workdir: Path, client: HoweClient, settings: Settings,
        should_continue: Callable[[], bool]) -> dict[str, Any]:
    started = time.monotonic()
    audio = workdir / "audio.wav"
    try:
        client.download(task_input["audio_url"], audio, 8 * 1024 * 1024)
    except DownloadRefused as exc:
        raise TaskFailed(exc.code, exc.retryable) from exc
    if not should_continue():
        raise Cancelled()

    # Le premier appel charge le modèle (plusieurs secondes) ; les suivants le trouvent en mémoire.
    with step("model", loaded=_model is not None):
        whisper = model(settings)
    samples = read_wav(audio)
    deadline = time.monotonic() + float(task_input.get("timeout") or 360) - 20
    waited = time.monotonic()
    with _model_lock:  # un seul passage à la fois dans le modèle chargé
        # Le journal compte les segments et la langue, jamais les paroles.
        with step("transcribe", audio_seconds=round(len(samples) / 16000, 1), lock_wait_ms=int((time.monotonic() - waited) * 1000),
                  device=settings.whisper_device) as noted:
            segments, info = whisper.transcribe(samples, beam_size=5, vad_filter=True, condition_on_previous_text=False)
            result = []
            for segment in segments:  # les segments arrivent au fil du calcul : on peut s'arrêter entre deux
                if not should_continue():
                    raise Cancelled()
                if time.monotonic() > deadline:
                    raise TaskFailed("asr_timeout", True)
                text = segment.text.strip()
                if text:
                    result.append({"start_ms": int(segment.start * 1000), "end_ms": int(segment.end * 1000), "text": text[:1000]})
            noted.update(segments=len(result), language=info.language)

    try:
        from faster_whisper import __version__ as version
    except ImportError:
        version = "?"
    return {
        "language": info.language,
        "model": settings.whisper_model,
        "segments": result,
        "engines": [f"faster-whisper {version}"],
        "metrics": {"duration_ms": int((time.monotonic() - started) * 1000)},
    }
