"""Le travailleur sur RunPod Serverless (lot 6, 13/11/2026).

Howé réveille l'endpoint (POST /run, entrée vide) quand des tâches attendent. Le conteneur
fait alors exactement ce que fait le travailleur d'un serveur : il prend des tâches par
l'API signée de Howé, les traite, rend ses résultats, puis s'arrête quand il n'y a plus
rien à faire depuis IDLE_EXIT_SECONDS. RunPod coupe ensuite le conteneur et sa facturation.

Rien de l'annonce ne passe par RunPod : ni dans l'entrée du réveil, ni dans ce que rend
`handle`, qui ne donne que des nombres.

Ce module ne dépend pas du paquet `runpod` : il se teste sans lui. `handler.py`, à la
racine, ne fait que le brancher.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from . import VERSION, video_transcribe
from .client import HoweClient
from .config import ConfigError, Settings, load
from .logs import error, log, logger, stack, where
from .runner import Runner, startup_fields

# Valeurs par défaut sur RunPod, si l'endpoint ne les donne pas : un conteneur ne doit
# jamais tourner sans fin sur un GPU facturé à la seconde.
DEFAULT_IDLE_EXIT_SECONDS = 20
DEFAULT_MAX_RUN_SECONDS = 1800

# Le SDK RunPod règle les journaux sur INFO : httpx écrirait alors chaque adresse appelée,
# liens signés des photos compris (trouvé à l'essai du 13/11/2026), et la console RunPod les
# garderait. Jamais d'URL signée dans un journal (spécification, section 16).
QUIET_LOGGERS = ("httpx", "httpcore", "urllib3", "requests", "huggingface_hub", "faster_whisper")


def quiet_http_logs() -> None:
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def settings_for_runpod(env: dict[str, str] | None = None) -> Settings:
    import os

    env = dict(os.environ if env is None else env)
    env.setdefault("IDLE_EXIT_SECONDS", str(DEFAULT_IDLE_EXIT_SECONDS))
    env.setdefault("MAX_RUN_SECONDS", str(DEFAULT_MAX_RUN_SECONDS))
    settings = load(env)
    if settings.idle_exit_seconds < 1:
        raise ConfigError("IDLE_EXIT_SECONDS doit valoir au moins 1 sur RunPod.")
    return settings


def warm_up(settings: Settings) -> None:
    """Charge Whisper avant la première tâche, si ce conteneur transcrit. Ne fait jamais
    échouer le démarrage : un conteneur qui meurt au démarrage redémarre en boucle, GPU allumé."""
    if "asr" not in settings.capabilities:
        return
    log("worker.gpu", cuda_devices=cuda_devices(), device=settings.whisper_device)
    started = time.monotonic()
    try:
        video_transcribe.model(settings)
        log("worker.warmed", model=settings.whisper_model, device=settings.whisper_device, seconds=f"{time.monotonic() - started:.1f}")
    except Exception as exc:  # noqa: BLE001 — la première transcription le signalera à Howé
        error("worker.warm_failed", reason=type(exc).__name__, code=getattr(exc, "code", None), detail=getattr(exc, "detail", None),
              ms=int((time.monotonic() - started) * 1000))


def cuda_devices() -> int | None:
    """Le nombre de GPU vus par CTranslate2 (le moteur de Whisper) ; None s'il est absent."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 — absent hors de l'image RunPod
        return None


def handle(job: dict[str, Any], settings: Settings | None = None, client: HoweClient | None = None) -> dict[str, Any]:
    """Un réveil : traiter les tâches qui attendent, puis rendre un bilan chiffré. Ne lève jamais."""
    quiet_http_logs()
    try:
        settings = settings or settings_for_runpod()
    except ConfigError as exc:
        error("worker.config_error", reason=str(exc))
        return {"status": "error", "code": "config", "detail": str(exc)[:200]}

    runner = Runner(settings, client or HoweClient(settings))
    job_input = job.get("input") if isinstance(job.get("input"), dict) else {}
    log("worker.woken", job=str(job.get("id", ""))[:64], test=bool(job_input.get("test")) or None, **startup_fields(settings))
    started = time.monotonic()
    try:
        runner.run_forever()
    except Exception as exc:  # noqa: BLE001 — un bilan d'erreur plutôt qu'une exception chez RunPod
        error("worker.crashed", error=type(exc).__name__, where=where(exc))
        logger.error("Pile d'appels :\n%s", stack(exc))
        return {"status": "error", "code": "worker_error", "outcomes": dict(runner.outcomes)}

    log("worker.wake_done", seconds=int(time.monotonic() - started), outcomes=dict(runner.outcomes))
    return {"status": "ok", "version": VERSION, "outcomes": dict(runner.outcomes)}
