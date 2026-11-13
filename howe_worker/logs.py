"""Le journal du travailleur, comme celui d'ocr-worker (13/11/2026, demande de l'utilisateur) :
le module `logging` de Python, une ligne de texte lisible par événement, sur la sortie
d'erreur, que la console RunPod et journalctl affichent telle quelle :

    2026-11-13 19:19:54,120 INFO     howe_worker — Tâche 4c4c61b6 prise : video_prepare, bail jusqu'à 19:21:54
    2026-11-13 19:19:55,410 INFO     howe_worker — [CHRONO] 4c4c61b6 · téléchargement — 1,29 s (cumul 1,30 s) · 2 984 081 octets
    2026-11-13 19:20:03,002 INFO     howe_worker — [CHRONO] 4c4c61b6 · TOTAL 8,88 s — encodage 4,10 s (46 %) · dépôt 2,95 s (33 %) · …

Chaque tâche a son chronomètre (même principe que `chrono.py` d'ocr-worker) : une ligne par
étape à sa fin, puis un résumé trié par durée décroissante, même quand la tâche échoue — c'est
souvent l'étape trop longue qu'on cherche.

LOG_LEVEL : INFO par défaut (tout le détail, choix de l'utilisateur), WARNING ou ERROR.

Jamais de texte lu, de paroles, de contenu de QR code, de lien signé, de jeton de bail ni de clé.
Une erreur inattendue s'écrit avec sa pile d'appels mais sans son message, qui peut porter une
adresse signée.
"""

from __future__ import annotations

import logging
import os
import string
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

# Le même réglage qu'ocr-worker : niveau, format, sortie d'erreur.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").strip().upper() or "INFO",
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("howe_worker")

LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

# La tâche en cours dans ce fil : sa référence, son type et son chronomètre.
_context: ContextVar[dict[str, Any]] = ContextVar("howe_worker_log_context", default={})

# Les événements en phrases ; les champs non repris dans la phrase suivent en « clé=valeur ».
MESSAGES = {
    "worker.started": "Travailleur {version} démarré : Howé {howe}, travailleur {worker}…, compétences {capabilities}, places {slots}",
    "worker.woken": "Réveil RunPod {job} : travailleur {version}, Howé {howe}, compétences {capabilities}, places {slots}",
    "worker.container_started": "Conteneur démarré : travailleur {version}, Python {python}",
    "worker.gpu": "GPU vus par CTranslate2 : {cuda_devices} ; Whisper sur {device}",
    "worker.warmed": "Whisper {model} chargé sur {device} en {seconds} s",
    "worker.warm_failed": "Whisper non chargé au démarrage : {reason}",
    "worker.config_error": "Configuration invalide : {reason}",
    "worker.container_setup_failed": "Démarrage du conteneur en échec : {error}",
    "worker.refused": "Howé refuse le travailleur ({status}, {code}) : {hint}",
    "worker.howe_unavailable": "Howé injoignable ({reason}) : nouvel essai dans {retry_in_seconds} s",
    "worker.howe_back": "Howé répond de nouveau",
    "worker.waiting": "Aucune tâche depuis {idle_seconds} s ; {done} tâche(s) traitée(s) depuis le démarrage",
    "worker.idle_exit": "Plus de tâche depuis {idle_seconds} s : arrêt",
    "worker.max_run_reached": "Durée maximale atteinte ({max_run_seconds} s) : plus de nouvelle tâche, {running} en cours",
    "worker.draining": "Arrêt : {running} tâche(s) à finir",
    "worker.stopped": "Travailleur arrêté après {uptime_seconds} s : {outcomes}",
    "worker.wake_done": "Réveil terminé en {seconds} s : {outcomes}",
    "worker.crashed": "Le travailleur s'est arrêté sur une erreur inattendue : {error}",
    "worker.once": "Prise unique : {taken} tâche(s)",
    "tasks.leased": "{count} tâche(s) prise(s) : {kinds} (places libres {free})",
    "task.started": "Tâche {task} prise : {kind}, bail jusqu'à {lease_expires_at}",
    "task.finished": "Tâche {task} terminée : {kind}, {outcome}",
    "task.cancelled_by_howe": "Tâche {task} annulée par Howé : arrêt",
    "task.heartbeat_failed": "Tâche {task} : battement de cœur sans réponse ({reason})",
    "task.unreported": "Tâche {task} : résultat non rendu à Howé ({reason}) ; elle reviendra dans la file",
    "task.kind_unsupported": "Tâche {task} : type {kind} inconnu du travailleur {version}",
    "task.crashed": "Tâche {task} ({kind}) : erreur inattendue {error}",
    "howe.call": "Howé {method} {path} → {status} en {ms} ms",
    "file.download": "Fichier {name} lu : {bytes} octets en {ms} ms",
    "file.upload": "Fichier {name} déposé chez Howé : {bytes} octets → {status} en {ms} ms",
    "file.upload_refused": "Fichier {name} refusé par Howé : le bail n'est plus à nous",
    "frames.read": "{count} image(s) de la vidéo lue(s) : {lines} ligne(s) de texte, {codes} QR code(s)",
}

# Les étapes des tâches, en mots (chronomètre).
STEP_NAMES = {
    "open": "ouverture", "tesseract": "lecture Tesseract", "qr": "QR codes", "probe": "sondage ffprobe",
    "encode": "encodage", "poster": "aperçu", "audio": "son", "frames": "images", "upload": "dépôt chez Howé",
    "frame": "lecture des images", "model": "chargement de Whisper", "transcribe": "transcription",
    "download": "téléchargement", "report": "rendu à Howé",
}


def set_level(name: str) -> None:
    """Change le niveau (tests)."""
    logger.setLevel(LEVELS.get(name.strip().lower(), logging.INFO))


def level_name() -> str:
    return logging.getLevelName(logger.getEffectiveLevel())


@contextmanager
def bound(**fields: Any) -> Iterator[None]:
    """Pendant le bloc, chaque ligne porte ces champs (la tâche en cours, son type)."""
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def log(event: str, level: str = "info", **fields: Any) -> None:
    try:
        # La référence de la tâche en cours suit chaque ligne ; son type seulement si la phrase le demande.
        context = {k: v for k, v in _context.get().items() if k == "task" or (k == "kind" and "{kind}" in MESSAGES.get(event, ""))}
        values = {k: v for k, v in {**context, **fields}.items() if v is not None}
        logger.log(LEVELS.get(level, logging.INFO), "%s", render(event, values))
    except Exception:  # noqa: BLE001 — le journal ne doit jamais casser ce qu'il observe
        pass


def warning(event: str, **fields: Any) -> None:
    log(event, "warning", **fields)


def error(event: str, **fields: Any) -> None:
    log(event, "error", **fields)


def render(event: str, values: dict[str, Any]) -> str:
    """La phrase de l'événement, puis les champs qu'elle ne reprend pas."""
    template = MESSAGES.get(event)
    if template is None:
        used: set[str] = set()
        head = event
    else:
        used = {name for _, name, _, _ in string.Formatter().parse(template) if name}
        head = template.format_map(_Shown({k: _show(k, values[k]) for k in used if k in values}))
    rest = " · ".join(f"{k}={_show(k, v)}" for k, v in values.items() if k not in used)
    return f"{head} · {rest}" if rest else head


class Stopwatch:
    """Le chronomètre d'une tâche, sur le modèle de `Chrono` d'ocr-worker.

    Ne lève jamais : une mesure qui ferait échouer la tâche serait pire que pas de mesure.
    """

    def __init__(self, reference: str):
        self.reference = reference or "?"
        self.started = time.monotonic()
        # {étape: (secondes cumulées, en échec)}, dans l'ordre d'arrivée.
        self.steps: dict[str, tuple[float, bool]] = {}

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def note(self, name: str, seconds: float, failed: bool, extra: dict[str, Any] | None = None, quiet: bool = False) -> None:
        try:
            previous = self.steps.get(name)
            self.steps[name] = ((previous[0] if previous else 0.0) + seconds, failed or (previous[1] if previous else False))
            if not quiet:
                details = " · ".join(f"{k}={_show(k, v)}" for k, v in (extra or {}).items() if v is not None)
                logger.info("[CHRONO] %s · %s%s — %s s (cumul %s s)%s", self.reference, name, " EN ÉCHEC" if failed else "",
                            _seconds(seconds), _seconds(self.elapsed), f" · {details}" if details else "")
        except Exception:  # noqa: BLE001
            pass

    def summary(self) -> str:
        total = self.elapsed
        parts = [f"{name} {_seconds(seconds)} s ({seconds / total * 100 if total > 0 else 0:.0f} %){' EN ÉCHEC' if failed else ''}"
                 for name, (seconds, failed) in sorted(self.steps.items(), key=lambda item: -item[1][0])]
        return f"TOTAL {_seconds(total)} s" + (f" — {' · '.join(parts)}" if parts else "")

    def log_summary(self, suffix: str = "") -> None:
        try:
            logger.info("[CHRONO] %s · %s%s", self.reference, self.summary(), f" — {suffix}" if suffix else "")
        except Exception:  # noqa: BLE001
            pass


@contextmanager
def stopwatch(reference: str) -> Iterator[Stopwatch]:
    """Un chronomètre pour la tâche en cours : les étapes (`step`) s'y inscrivent."""
    watch = Stopwatch(reference)
    with bound(_stopwatch=watch):
        yield watch


@contextmanager
def step(name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Chronomètre une étape de la tâche en cours. Le dictionnaire rendu reçoit des mesures
    (tailles, nombres) écrites sur la ligne de l'étape. L'erreur passe ; la durée est notée.

    Une étape répétée (une par image d'une vidéo) se cumule : `quiet=True` ne garde que le cumul,
    écrit dans le résumé, au lieu d'une ligne par image.
    """
    quiet = bool(fields.pop("quiet", False))
    label = STEP_NAMES.get(name, name) + (f" ({fields.pop('variant')})" if "variant" in fields else "")
    extra: dict[str, Any] = dict(fields)
    watch: Stopwatch | None = _context.get().get("_stopwatch")
    started = time.monotonic()
    try:
        yield extra
    except BaseException:
        _note(watch, label, time.monotonic() - started, True, extra, False)
        raise
    _note(watch, label, time.monotonic() - started, False, extra, quiet)


def _note(watch: Stopwatch | None, label: str, seconds: float, failed: bool, extra: dict[str, Any], quiet: bool) -> None:
    if watch is None:
        # Hors d'une tâche (essais) : une ligne simple.
        watch = Stopwatch(str(_context.get().get("task", "?"))[:8])
        watch.started -= seconds
    watch.note(label, seconds, failed, extra, quiet)


def host_of(url: str) -> str:
    """L'hôte d'une adresse, sans chemin ni paramètres : jamais un lien signé entier."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}" if parts.netloc else "?"


def where(exc: BaseException, limit: int = 6) -> list[str]:
    """Où une erreur est née : « fichier.py:ligne fonction », du plus profond au moins profond."""
    frames = traceback.extract_tb(exc.__traceback__)
    return [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in reversed(frames)][:limit]


def stack(exc: BaseException) -> str:
    """La pile d'appels d'une erreur, comme `traceback.format_exc()` d'ocr-worker, mais terminée
    par le seul type de l'erreur : son message peut contenir une adresse signée."""
    return "".join(traceback.format_tb(exc.__traceback__)) + type(exc).__name__


def _show(key: str, value: Any) -> str:
    if key == "task" and isinstance(value, str) and len(value) == 36:
        return value[:8]
    if key == "bytes" and isinstance(value, int):
        return f"{value:,}".replace(",", " ")
    if isinstance(value, dict):
        return ", ".join(f"{k} {v}" for k, v in value.items()) or "aucun"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "aucun"
    return str(value)


def _seconds(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


class _Shown(dict):
    def __missing__(self, key: str) -> str:
        return "?"
