"""Réglages du travailleur, lus dans l'environnement (spécification, section 15).

Variables :
- HOWE_URL : adresse de Howé, par exemple https://howe.immo
- WORKER_REFERENCE : la référence donnée par « php artisan moderation:worker-add »
- WORKER_PRIVATE_KEY_FILE : fichier (0600) qui contient la clé privée en base64 ; à défaut,
  WORKER_PRIVATE_KEY, la clé elle-même : sur RunPod, un secret de l'endpoint (lot 6), jamais
  écrit dans le dépôt ni dans l'image ; ailleurs, seulement pour les essais
- CAPABILITIES : compétences annoncées, séparées par des virgules (image au lot 3b)
- SLOTS : places par compétence, par exemple « image=3 »
- TMP_DIR : dossier des fichiers temporaires (un sous-dossier par tâche, effacé à la fin)
- TESSERACT_CMD, LEASE_WAIT, HEARTBEAT_SECONDS, MAX_DOWNLOAD_MB, TASK_MEMORY_MB, TASK_CPU_SECONDS
- lot 4b : FFMPEG_CMD, FFPROBE_CMD, VIDEO_MEMORY_MB, WHISPER_MODEL (small), ASR_THREADS,
  WHISPER_DOWNLOAD=1 pour permettre le téléchargement du modèle (sinon : seulement le cache)
- lot 6 (RunPod Serverless) : WHISPER_DEVICE (cpu, cuda ou auto) et WHISPER_COMPUTE_TYPE (int8 sur
  processeur, float16 sur GPU) ; IDLE_EXIT_SECONDS (s'arrêter après tant de secondes sans tâche,
  0 = jamais) et MAX_RUN_SECONDS (ne plus prendre de tâche passé ce délai, 0 = sans limite)
- LOG_LEVEL : info (par défaut : tout le détail), warning ou error, lu par howe_worker/logs.py
"""

from __future__ import annotations

import base64
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(Exception):
    """Réglage absent ou invalide : le travailleur ne démarre pas."""


@dataclass(frozen=True)
class Settings:
    howe_url: str
    reference: str
    seed: bytes = field(repr=False)
    capabilities: tuple[str, ...] = ("image",)
    slots: dict[str, int] = field(default_factory=lambda: {"image": 1})
    tmp_dir: Path = Path("/tmp/howe-moderation-worker")
    tesseract_cmd: str = "tesseract"
    lease_wait: int = 20
    heartbeat_seconds: int = 20
    max_download_bytes: int = 30 * 1024 * 1024
    task_memory_mb: int = 1024
    task_cpu_seconds: int = 120
    # Lot 4b : la vidéo (FFmpeg) et la transcription (Faster-Whisper).
    ffmpeg_cmd: str = "ffmpeg"
    ffprobe_cmd: str = "ffprobe"
    video_memory_mb: int = 3072
    whisper_model: str = "small"
    whisper_threads: int = 2
    # Le modèle est déjà sur le serveur (ou dans le cache) : jamais de téléchargement pendant une tâche.
    whisper_local_only: bool = True
    # Lot 6 : sur RunPod, Whisper tourne sur le GPU, et le conteneur s'arrête quand il n'y a plus rien à faire.
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    idle_exit_seconds: int = 0
    max_run_seconds: int = 0

    @property
    def total_slots(self) -> int:
        return sum(self.slots.get(c, 0) for c in self.capabilities)


def _int(env: dict[str, str], name: str, default: int, minimum: int = 0) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} doit être un nombre entier.") from exc
    if value < minimum:
        raise ConfigError(f"{name} doit valoir au moins {minimum}.")
    return value


def _choice(env: dict[str, str], name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = env.get(name, "").strip() or default
    if value not in allowed:
        raise ConfigError(f"{name} : {', '.join(allowed)}.")
    return value


def _seed(env: dict[str, str]) -> bytes:
    path = env.get("WORKER_PRIVATE_KEY_FILE", "").strip()
    if path:
        key_file = Path(path)
        if not key_file.is_file():
            raise ConfigError("WORKER_PRIVATE_KEY_FILE ne désigne pas un fichier.")
        mode = key_file.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise ConfigError("Le fichier de clé privée doit être en 0600 (lisible par son seul propriétaire).")
        raw = key_file.read_text().strip()
    else:
        raw = env.get("WORKER_PRIVATE_KEY", "").strip()
    if not raw:
        raise ConfigError("Clé privée absente : WORKER_PRIVATE_KEY_FILE.")
    try:
        seed = base64.b64decode(raw, validate=True)
    except ValueError as exc:
        raise ConfigError("Clé privée illisible (base64 attendu).") from exc
    if len(seed) != 32:
        raise ConfigError("Clé privée invalide : 32 octets attendus.")
    return seed


def load(env: dict[str, str] | None = None) -> Settings:
    env = dict(os.environ if env is None else env)
    howe_url = env.get("HOWE_URL", "").strip().rstrip("/")
    if not howe_url.startswith(("https://", "http://")):
        raise ConfigError("HOWE_URL doit commencer par https:// (http:// seulement en local).")
    reference = env.get("WORKER_REFERENCE", "").strip()
    if len(reference) != 36:
        raise ConfigError("WORKER_REFERENCE attendue (36 caractères).")

    capabilities = tuple(c.strip() for c in env.get("CAPABILITIES", "image").split(",") if c.strip())
    unknown = set(capabilities) - {"image", "video", "asr"}
    if not capabilities or unknown:
        raise ConfigError("CAPABILITIES : image, video ou asr.")
    slots: dict[str, int] = {}
    for part in env.get("SLOTS", "image=1").split(","):
        if not part.strip():
            continue
        name, _, count = part.partition("=")
        try:
            slots[name.strip()] = max(0, int(count))
        except ValueError as exc:
            raise ConfigError("SLOTS : par exemple « image=3 ».") from exc

    settings = Settings(
        howe_url=howe_url,
        reference=reference,
        seed=_seed(env),
        capabilities=capabilities,
        slots=slots,
        tmp_dir=Path(env.get("TMP_DIR", "/tmp/howe-moderation-worker")),
        tesseract_cmd=env.get("TESSERACT_CMD", "tesseract"),
        lease_wait=_int(env, "LEASE_WAIT", 20),
        heartbeat_seconds=_int(env, "HEARTBEAT_SECONDS", 20, 1),
        max_download_bytes=_int(env, "MAX_DOWNLOAD_MB", 30, 1) * 1024 * 1024,
        task_memory_mb=_int(env, "TASK_MEMORY_MB", 1024, 128),
        task_cpu_seconds=_int(env, "TASK_CPU_SECONDS", 120, 5),
        ffmpeg_cmd=env.get("FFMPEG_CMD", "ffmpeg"),
        ffprobe_cmd=env.get("FFPROBE_CMD", "ffprobe"),
        video_memory_mb=_int(env, "VIDEO_MEMORY_MB", 3072, 512),
        whisper_model=env.get("WHISPER_MODEL", "small"),
        whisper_threads=_int(env, "ASR_THREADS", 2, 1),
        whisper_local_only=env.get("WHISPER_DOWNLOAD", "0") != "1",
        whisper_device=_choice(env, "WHISPER_DEVICE", "cpu", ("cpu", "cuda", "auto")),
        whisper_compute_type=_choice(env, "WHISPER_COMPUTE_TYPE", "int8", ("int8", "int8_float16", "float16", "float32", "default")),
        idle_exit_seconds=_int(env, "IDLE_EXIT_SECONDS", 0),
        max_run_seconds=_int(env, "MAX_RUN_SECONDS", 0),
    )
    if settings.total_slots < 1:
        raise ConfigError("Aucune place : SLOTS doit en donner au moins une à une compétence annoncée.")
    return settings
