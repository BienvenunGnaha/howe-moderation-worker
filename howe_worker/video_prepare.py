"""video_prepare : contrôle et préparation d'une vidéo d'annonce (spécification,
sections 3.3, 12 et 13 ; lot 4b).

1. L'original est téléchargé par le lien signé, puis son vrai type est vérifié
   (MP4/MOV « ftyp », WebM « EBML ») : une liste de lecture HLS déguisée est refusée.
2. ffprobe (10 s au plus, fichiers locaux seulement, formats vidéo seulement) lit la
   durée et les flux. Plus de 45,5 s, aucune image, ou illisible : vidéo refusée.
3. FFmpeg encode une seule fois la copie publiable : H.264 High, 1 280 px au plus,
   30 i/s au plus, CRF 23 plafonné à 2,5 Mbit/s, AAC 96 kbit/s, faststart, sans
   métadonnées (position GPS, modèle du téléphone). Elle n'est publiée par Howé
   qu'à l'approbation.
4. Image d'aperçu WebP prise à 1 s ; son en WAV mono 16 kHz pour la transcription.
5. Images : deux par seconde, plus chaque changement de plan net ; les images presque
   identiques sont écartées par une empreinte perceptuelle (dHash).
6. Les fichiers sont déposés chez Howé par l'API signée ; rien ne reste sur le travailleur.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from . import logs
from .client import DownloadRefused, HoweClient, LeaseLost
from .config import Settings
from .photo_check import Cancelled, ProcessFailed, TaskFailed, run_limited

# Seuls les démultiplexeurs des formats acceptés par Howé ; ni HLS, ni concat, ni URL.
FORMATS = "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm"
SAFE_INPUT = ["-protocol_whitelist", "file", "-format_whitelist", FORMATS]
# Entrée de FFmpeg : 2 fils de décodage, quel que soit le nombre de cœurs. Sans cela, FFmpeg en lance
# un par cœur, et sur une machine RunPod (32 cœurs et plus) il dépasse le plafond de mémoire de la
# tâche et échoue : c'était le « video_unreadable » des vidéos de l'annonce 38 (13/11/2026).
THREADS = "2"
FFMPEG_INPUT = [*SAFE_INPUT, "-threads", THREADS]


def _check_signature(path: Path) -> None:
    with path.open("rb") as handle:
        head = handle.read(16)
    if head[4:8] != b"ftyp" and not head.startswith(b"\x1a\x45\xdf\xa3"):
        raise TaskFailed("file_unsupported", False, "pas une vidéo MP4, MOV ou WebM")


def probe(path: Path, settings: Settings, should_continue: Callable[[], bool]) -> dict[str, Any]:
    """Durée, dimensions et flux, lus par ffprobe. Illisible : refus (jamais une panne)."""
    try:
        out = run_limited([settings.ffprobe_cmd, "-v", "error", *SAFE_INPUT, "-show_streams", "-show_format", "-of", "json", str(path)],
                          settings, should_continue, timeout=10, memory_mb=settings.video_memory_mb, failure_code="probe_failed")
        data = json.loads(out)
    except ProcessFailed as exc:
        if exc.killed:
            raise TaskFailed("probe_failed", True, exc.detail) from exc
        raise TaskFailed("video_unreadable", False, f"ffprobe {exc.detail}") from exc
    except TaskFailed as exc:
        if exc.code == "probe_timeout":
            raise TaskFailed("video_unreadable", False, "ffprobe trop long") from exc
        raise
    except json.JSONDecodeError as exc:
        raise TaskFailed("video_unreadable", False) from exc

    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    try:
        duration = float((data.get("format") or {}).get("duration") or (video or {}).get("duration") or 0)
    except ValueError:
        duration = 0.0
    return {"video": video, "has_audio": audio is not None, "duration": duration}


def ffmpeg(args: list[str], settings: Settings, should_continue: Callable[[], bool], timeout: float,
           stderr_file: Path | None = None) -> None:
    # Sans fichier pour sa sortie d'erreur, FFmpeg n'écrit que ses erreurs (jamais les métadonnées
    # du fichier, position GPS du téléphone comprise) ; elles expliquent un échec.
    quiet = [] if stderr_file else ["-v", "error"]
    # 2 fils aussi pour les filtres et l'encodage de la sortie (le dernier argument est le fichier produit).
    *options, output = args
    try:
        run_limited([settings.ffmpeg_cmd, "-hide_banner", "-nostdin", "-y", *quiet, "-filter_threads", THREADS, *options,
                     "-threads", THREADS, output],
                    settings, should_continue, timeout=timeout, stderr_file=stderr_file, memory_mb=settings.video_memory_mb,
                    failure_code="encode_failed")
    except ProcessFailed as exc:
        # Tué (mémoire, temps) : panne, réessayée. Sorti en erreur : la vidéo n'a pas pu être lue.
        raise TaskFailed("encode_failed", True, exc.detail) if exc.killed else TaskFailed("video_unreadable", False, f"ffmpeg {exc.detail}") from exc


def dhash(path: Path) -> int:
    """Empreinte perceptuelle : deux images presque identiques ont presque la même."""
    small = Image.open(path).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return bits


def extract_frames(published: Path, workdir: Path, settings: Settings, should_continue: Callable[[], bool],
                   options: dict[str, Any]) -> list[dict[str, Any]]:
    """Deux images par seconde et chaque changement de plan, sans les doublons."""
    raw = workdir / "raw"
    raw.mkdir()
    log = workdir / "frames.log"
    step = 1 / max(float(options.get("per_second") or 2), 0.1)
    scene = float(options.get("scene") or 0.3)
    select = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{step:.3f})+gt(scene\\,{scene})',showinfo"
    ffmpeg([*FFMPEG_INPUT, "-i", str(published), "-vf", select, "-fps_mode", "vfr", "-q:v", "3", str(raw / "r%04d.jpg")],
           settings, should_continue, timeout=90, stderr_file=log)

    times = [float(t) for t in re.findall(r"pts_time:\s*([0-9.]+)", log.read_text(errors="replace"))]
    kept: list[dict[str, Any]] = []
    previous: int | None = None
    limit = int(options.get("max") or 120)
    (workdir / "frames").mkdir()
    for index, source in enumerate(sorted(raw.glob("r*.jpg"))):
        signature = dhash(source)
        if previous is not None and bin(signature ^ previous).count("1") <= 5:
            continue  # presque la même que la précédente gardée
        previous = signature
        name = f"frames/f{len(kept) + 1:04d}.jpg"
        source.rename(workdir / name)
        kept.append({"name": name, "at_ms": int(round((times[index] if index < len(times) else index * step) * 1000))})
        if len(kept) >= limit:
            break
    return kept


def run(task_input: dict[str, Any], workdir: Path, client: HoweClient, settings: Settings,
        should_continue: Callable[[], bool], task: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.monotonic()
    original = workdir / "original"
    try:
        client.download(task_input["read_url"], original, settings.max_download_bytes)
    except DownloadRefused as exc:
        raise TaskFailed(exc.code, exc.retryable) from exc
    _check_signature(original)

    with logs.step("probe") as noted:
        info = probe(original, settings, should_continue)
        noted.update(seconds=round(info["duration"], 1), has_audio=info["has_audio"],
                     width=(info["video"] or {}).get("width"), height=(info["video"] or {}).get("height"))
    if info["video"] is None:
        raise TaskFailed("video_no_stream", False, "aucune image dans le fichier")
    limit = float((task_input.get("limits") or {}).get("max_video_seconds") or 45.5)
    if info["duration"] > limit:
        raise TaskFailed("video_too_long", False, f"{info['duration']:.1f} s")
    if not should_continue():
        raise Cancelled()

    publish = task_input.get("publish") or {}
    side = int(publish.get("max_side") or 1280)
    published = workdir / "published.mp4"
    scale = f"scale=w='min({side},iw)':h='min({side},ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,format=yuv420p"
    encode_timeout = max(30.0, float(task_input.get("timeout") or 240) - 60)
    with logs.step("encode") as noted:
        ffmpeg([*FFMPEG_INPUT, "-i", str(original), "-map", "0:v:0", "-map", "0:a:0?", "-vf", scale,
                "-fpsmax", str(publish.get("max_fps") or 30), "-c:v", "libx264", "-profile:v", "high", "-preset", "veryfast",
                "-crf", str(publish.get("crf") or 23), "-maxrate", str(publish.get("max_bitrate") or "2500k"), "-bufsize", "5000k",
                "-c:a", "aac", "-b:a", str(publish.get("audio_bitrate") or "96k"), "-ac", "2",
                "-movflags", "+faststart", "-map_metadata", "-1", "-map_chapters", "-1", "-sn", "-dn",
                str(published)], settings, should_continue, timeout=encode_timeout)
        noted.update(bytes_in=original.stat().st_size, bytes_out=published.stat().st_size)

    poster_png = workdir / "poster.png"
    at = min(float(publish.get("poster_at") or 1.0), max(info["duration"] / 2, 0))
    with logs.step("poster"):
        ffmpeg(["-ss", f"{at:.2f}", *FFMPEG_INPUT, "-i", str(published), "-frames:v", "1", str(poster_png)], settings, should_continue, timeout=30)
        poster = workdir / "poster.webp"
        Image.open(poster_png).convert("RGB").save(poster, "WEBP", quality=80)

    files = ["published.mp4", "poster.webp"]
    if info["has_audio"]:
        with logs.step("audio"):
            ffmpeg([*FFMPEG_INPUT, "-i", str(original), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                    str(workdir / "audio.wav")], settings, should_continue, timeout=60)
        files.append("audio.wav")

    with logs.step("frames") as noted:
        frames = extract_frames(published, workdir, settings, should_continue, task_input.get("frames") or {})
        noted["kept"] = len(frames)
    files += [frame["name"] for frame in frames]
    out = probe(published, settings, should_continue)["video"] or {}

    if task is not None:
        with logs.step("upload") as noted:
            noted["files"] = len(files)
            for name in files:
                if not should_continue():
                    raise Cancelled()
                try:
                    client.upload(task["reference"], task["lease_token"], name, workdir / name)
                except LeaseLost as exc:
                    logs.warning("file.upload_refused", name=name)
                    raise Cancelled() from exc

    return {
        "duration_ms": int(round(info["duration"] * 1000)),
        "width": out.get("width"),
        "height": out.get("height"),
        "has_audio": info["has_audio"],
        "frames": frames,
        "files": files,
        "engines": [ffmpeg_version(settings)],
        "metrics": {"duration_ms": int((time.monotonic() - started) * 1000), "frames_read": len(frames)},
    }


def ffmpeg_version(settings: Settings) -> str:
    import subprocess

    try:
        first = subprocess.run([settings.ffmpeg_cmd, "-version"], capture_output=True, text=True, timeout=10).stdout.split("\n")[0]
        return " ".join(first.split()[:3]) or "ffmpeg"
    except (OSError, subprocess.SubprocessError):
        return "ffmpeg"
