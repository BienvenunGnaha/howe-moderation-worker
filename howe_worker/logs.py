"""Journal JSON sur la sortie standard (spécification, section 16) : une ligne par
événement, avec la référence de la tâche, son type, ses durées et ses mesures.

Jamais de texte lu, de contenu de fichier, de lien signé ni de clé.

Niveaux (LOG_LEVEL, « info » par défaut, choix de l'utilisateur du 13/11/2026) :
- info : tout le détail, la vie du travailleur et de chaque tâche, chaque appel à Howé et
  chaque étape d'une tâche avec sa durée (téléchargement, lecture Tesseract, encodage,
  transcription…) ;
- warning : seulement les avertissements et les erreurs ;
- error : seulement les erreurs.

Chaque ligne porte « level ».
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

LEVELS = {"info": 20, "warning": 30, "error": 40}

_threshold = LEVELS.get(os.environ.get("LOG_LEVEL", "info").strip().lower(), LEVELS["info"])
_write_lock = threading.Lock()
# La tâche en cours dans ce fil : chaque ligne écrite pendant la tâche porte sa référence et son type.
_context: ContextVar[dict[str, Any]] = ContextVar("howe_worker_log_context", default={})


@contextmanager
def bound(**fields: Any) -> Iterator[None]:
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def set_level(name: str) -> None:
    """Change le niveau (tests, ou LOG_LEVEL relu par la configuration)."""
    global _threshold
    _threshold = LEVELS.get(name.strip().lower(), LEVELS["info"])


def level_name() -> str:
    return next(name for name, value in LEVELS.items() if value == _threshold)


def log(event: str, level: str = "info", **fields: Any) -> None:
    if LEVELS.get(level, LEVELS["info"]) < _threshold:
        return
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": level, "event": event}
    record.update({k: v for k, v in {**_context.get(), **fields}.items() if v is not None})
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    # Plusieurs tâches tournent en parallèle : une ligne ne doit jamais se couper en deux.
    with _write_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def warning(event: str, **fields: Any) -> None:
    log(event, "warning", **fields)


def error(event: str, **fields: Any) -> None:
    log(event, "error", **fields)


@contextmanager
def step(name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Chronomètre une étape d'une tâche. Le dictionnaire rendu reçoit des mesures à ajouter
    à la ligne (tailles, nombres). Une étape qui lève est notée « échouée » avec le type de
    l'erreur, puis l'erreur continue son chemin."""
    extra: dict[str, Any] = {}
    started = time.monotonic()
    try:
        yield extra
    except BaseException as exc:
        log("step.failed", step=name, ms=_ms(started), error=type(exc).__name__, **fields, **extra)
        raise
    log("step.done", step=name, ms=_ms(started), **fields, **extra)


def host_of(url: str) -> str:
    """L'hôte d'une adresse, sans chemin ni paramètres : jamais un lien signé entier."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}" if parts.netloc else "?"


def where(exc: BaseException, limit: int = 6) -> list[str]:
    """Où une erreur est née : « fichier.py:ligne fonction », du plus profond au moins profond.

    Sans le message de l'erreur, qui peut contenir une adresse (lien signé) ou du texte lu.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    return [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in reversed(frames)][:limit]


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
