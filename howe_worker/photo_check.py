"""photo_check : le texte visible et les QR codes d'une photo (spécification,
sections 3.2, 12 et 13).

1. Le fichier est téléchargé par le lien signé de la tâche, taille plafonnée.
2. Son type est vérifié par son contenu (signature du fichier), jamais par son nom ;
   ses dimensions sont lues avant tout décodage (bombe de décompression).
3. Tesseract (fra, eng) lit deux versions : l'originale, et une version en niveaux
   de gris contrastée, agrandie si elle est petite (panneaux « À louer »).
4. ZBar lit les QR codes ; OpenCV relit ceux que ZBar a manqués.

Le travailleur rend les lignes lues et leur confiance : Howé leur applique ses
règles (choix du 12/11/2026). Rien de lu n'est écrit dans un journal.
"""

from __future__ import annotations

import csv
import io
import os
import resource
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps

from . import measure
from .client import DownloadRefused, HoweClient
from .config import Settings
from .logs import step
from .measure import MeasuredPopen

MAX_LINES = 2000
MAX_LINE_LENGTH = 500

# Signatures des formats acceptés par Howé à l'envoi (JPEG, PNG, WebP ; GIF par prudence).
SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a")


class TaskFailed(Exception):
    """Échec rendu à Howé par /fail : retryable ou non (fichier refusé)."""

    def __init__(self, code: str, retryable: bool, detail: str | None = None):
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.detail = detail


class Cancelled(Exception):
    """Howé a annulé la tâche (battement de cœur « continue : faux ») : on s'arrête."""


def _limits(memory_mb: int, cpu_seconds: int) -> Callable[[], None]:
    def apply() -> None:
        # Sous-processus plafonné en mémoire, en temps processeur et en taille de fichier écrit.
        resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 * 1024,) * 2)
        os.nice(10)

    return apply


def run_limited(command: list[str], settings: Settings, should_continue: Callable[[], bool], timeout: float = 60.0,
                stderr_file: Path | None = None, memory_mb: int | None = None, failure_code: str = "ocr_failed") -> str:
    """Lance un programme limité, et le tue si la tâche est annulée ou trop longue.

    stderr_file : où écrire sa sortie d'erreur (FFmpeg y donne l'instant de chaque image).
    Sans lui, la sortie d'erreur est gardée en mémoire : ses dernières lignes expliquent un échec.
    """
    stderr = stderr_file.open("wb") if stderr_file else subprocess.PIPE
    process = MeasuredPopen(
        command,
        stdout=subprocess.PIPE,
        stderr=stderr,
        preexec_fn=_limits(memory_mb or settings.task_memory_mb, settings.task_cpu_seconds),
        # MALLOC_ARENA_MAX : sans lui, chaque fil d'exécution réserve sa propre zone de mémoire ; sur une
        # machine à 32 cœurs et plus (RunPod), FFmpeg dépasse le plafond d'adresses et échoue (13/11/2026).
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "OMP_THREAD_LIMIT": "1", "LC_ALL": "C.UTF-8", "MALLOC_ARENA_MAX": "2"},
    )
    try:
        return _wait(process, command, should_continue, timeout, failure_code, stderr_file)
    finally:
        # Mesure de la tâche (section 16), même pour un programme arrêté ou en échec.
        measure.note_tool(process.max_rss_kb)
        if stderr_file:
            stderr.close()


def _wait(process: subprocess.Popen, command: list[str], should_continue: Callable[[], bool], timeout: float, failure_code: str,
          stderr_file: Path | None = None) -> str:
    deadline = time.monotonic() + timeout
    while True:
        try:
            out, err = process.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if not should_continue():
                process.kill()
                process.communicate()
                raise Cancelled()
            if time.monotonic() > deadline:
                process.kill()
                process.communicate()
                raise TaskFailed(failure_code.replace("_failed", "_timeout"), True)
    if process.returncode != 0:
        # Tué par un signal (mémoire ou temps processeur épuisés) : panne, réessayée.
        # Sorti en erreur : le programme n'a pas su lire le fichier.
        raise ProcessFailed(failure_code, process.returncode, _error_tail(err, stderr_file))
    return out.decode("utf-8", errors="replace")


def _error_tail(err: bytes | None, stderr_file: Path | None) -> str:
    """Les deux dernières lignes d'erreur d'un programme, pour comprendre un échec.

    FFmpeg et ffprobe tournent en « -v error » (sans les métadonnées du fichier, position GPS
    comprise) ; quand la sortie va dans un fichier en niveau normal (images d'une vidéo), seules
    les lignes qui parlent d'erreur sont gardées.
    """
    if stderr_file is not None:
        try:
            lines = [l for l in stderr_file.read_text(errors="replace").splitlines() if "error" in l.lower()]
        except OSError:
            lines = []
    else:
        lines = (err or b"").decode("utf-8", errors="replace").splitlines()
    lines = [" ".join(l.split()) for l in lines if l.strip()]
    return " | ".join(lines[-2:])[:180]


class ProcessFailed(TaskFailed):
    """Un programme sorti en erreur ; returncode négatif = tué par un signal (mémoire, temps).

    Le détail porte le code de sortie et les dernières lignes d'erreur du programme.
    """

    def __init__(self, code: str, returncode: int, errors: str = ""):
        super().__init__(code, True, f"code {returncode}" + (f" : {errors}" if errors else ""))
        self.returncode = returncode

    @property
    def killed(self) -> bool:
        return self.returncode < 0


def tesseract_lines(path: Path, version: str, settings: Settings, should_continue: Callable[[], bool],
                    timeout: float = 30.0, psm: int = 3) -> list[dict[str, Any]]:
    """Les lignes lues par Tesseract, chacune avec la confiance moyenne de ses mots.

    Découpage automatique de la page (psm 3) : mesuré le 13/11/2026, il lit aussi bien
    un panneau encadré que des mots épars dans le décor, alors que le mode « texte
    épars » (psm 11) manque les panneaux encadrés.
    """
    tsv = run_limited([settings.tesseract_cmd, str(path), "stdout", "-l", "fra+eng", "--psm", str(psm), "tsv"],
                      settings, should_continue, timeout)
    lines: dict[tuple[str, str, str, str], list[tuple[str, float]]] = {}
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (row.get("text") or "").strip()
        try:
            confidence = float(row.get("conf") or -1)
        except ValueError:
            continue
        if not text or confidence < 0:
            continue
        key = (row.get("page_num", ""), row.get("block_num", ""), row.get("par_num", ""), row.get("line_num", ""))
        lines.setdefault(key, []).append((text, confidence))
    result = []
    for words in lines.values():
        text = " ".join(w for w, _ in words)[:MAX_LINE_LENGTH]
        result.append({"text": text, "confidence": round(sum(c for _, c in words) / len(words), 1), "version": version})
    return result


def tesseract_version(settings: Settings) -> str:
    try:
        out = subprocess.run([settings.tesseract_cmd, "--version"], capture_output=True, text=True, timeout=10).stdout
        return out.splitlines()[0].strip() if out else "tesseract"
    except (OSError, subprocess.SubprocessError):
        return "tesseract"


def read_qr(images: list[Image.Image]) -> tuple[list[dict[str, str]], list[str]]:
    """Les QR codes, lus par ZBar, puis par OpenCV si ZBar n'a rien vu."""
    found: dict[str, dict[str, str]] = {}
    engines: list[str] = []
    try:
        from pyzbar import pyzbar

        engines.append("zbar")
        for image in images:
            for symbol in pyzbar.decode(image, symbols=[pyzbar.ZBarSymbol.QRCODE]):
                data = symbol.data.decode("utf-8", errors="replace")
                found.setdefault(data, {"data": data, "engine": "zbar"})
    except ImportError:
        pass
    if not found:
        try:
            import cv2
            import numpy as np

            engines.append(f"opencv {cv2.__version__}")
            detector = cv2.QRCodeDetector()
            for image in images:
                array = np.array(image.convert("L"))
                ok, decoded, _, _ = detector.detectAndDecodeMulti(array)
                if ok:
                    for data in decoded:
                        if data:
                            found.setdefault(data, {"data": data, "engine": "opencv"})
        except ImportError:
            pass
    return list(found.values())[:20], engines


def open_checked(path: Path, settings: Settings, limits: dict[str, Any]) -> Image.Image:
    """Ouvre l'image après avoir vérifié son type réel et ses dimensions."""
    with path.open("rb") as handle:
        head = handle.read(16)
    webp = head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    if not (webp or any(head.startswith(s) for s in SIGNATURES)):
        raise TaskFailed("file_unsupported", False, "type de fichier refusé")

    max_pixels = int(limits.get("max_pixels") or 40_000_000)
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        image = Image.open(path)
        width, height = image.size  # lu dans l'en-tête, sans décoder
        if width * height > max_pixels:
            raise TaskFailed("image_too_large", False, f"{width}x{height}")
        image = ImageOps.exif_transpose(image)
        image.load()
    except TaskFailed:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise TaskFailed("image_too_large", False)
    except (OSError, SyntaxError, ValueError) as exc:
        raise TaskFailed("file_unreadable", False, type(exc).__name__)

    image = image.convert("RGB")
    max_side = int(limits.get("max_side") or 3840)
    if max(image.size) > max_side:
        # Une photo de téléphone dépasse souvent 3 840 px : réduite pour la lecture, jamais refusée pour ça.
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def run(task_input: dict[str, Any], workdir: Path, client: HoweClient, settings: Settings,
        should_continue: Callable[[], bool]) -> dict[str, Any]:
    started = time.monotonic()
    original = workdir / "original"
    try:
        client.download(task_input["read_url"], original, settings.max_download_bytes)
    except DownloadRefused as exc:
        raise TaskFailed(exc.code, exc.retryable) from exc
    if not should_continue():
        raise Cancelled()

    with step("open") as noted:
        image = open_checked(original, settings, task_input.get("limits") or {})
        width, height = image.size
        noted.update(width=width, height=height)
    plain = workdir / "plain.png"
    image.save(plain)

    # Version contrastée : niveaux de gris, contraste étiré, agrandie si petite, un peu plus nette.
    gray = ImageOps.autocontrast(ImageOps.grayscale(image), cutoff=2)
    if gray.width < 1600:
        factor = min(2.0, 1600 / gray.width)
        gray = gray.resize((int(gray.width * factor), int(gray.height * factor)), Image.Resampling.LANCZOS)
    gray = gray.filter(ImageFilter.SHARPEN)
    contrast = workdir / "contrast.png"
    gray.save(contrast)

    # Le délai de la tâche (60 s) se partage entre les deux lectures.
    timeout = max(5.0, float(task_input.get("timeout") or 60) / 2)
    # Le journal compte les lignes lues, jamais leur texte.
    with step("tesseract", variant="original") as noted:
        lines = tesseract_lines(plain, "original", settings, should_continue, timeout, psm=3)
        noted["lines"] = len(lines)
    if not should_continue():
        raise Cancelled()
    with step("tesseract", variant="contrast") as noted:
        more = tesseract_lines(contrast, "contrast", settings, should_continue, timeout, psm=3)
        noted["lines"] = len(more)
    lines += more
    if not should_continue():
        raise Cancelled()

    with step("qr") as noted:
        qr, qr_engines = read_qr([image, gray])
        noted["codes"] = len(qr)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)

    return {
        "text": lines[:MAX_LINES],
        "qr": qr,
        "engines": [tesseract_version(settings), *qr_engines],
        "metrics": {
            "duration_ms": int((time.monotonic() - started) * 1000),
            "memory_mb": int(usage.ru_maxrss / 1024),
            "width": width,
            "height": height,
        },
    }
