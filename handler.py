"""Point d'entrée RunPod Serverless. La logique est dans howe_worker/serverless.py (README, lot 6)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runpod  # noqa: E402

from howe_worker import VERSION  # noqa: E402
from howe_worker.config import ConfigError  # noqa: E402
from howe_worker.logs import error, log, logger, stack  # noqa: E402
from howe_worker.serverless import handle, quiet_http_logs, settings_for_runpod, warm_up  # noqa: E402

quiet_http_logs()
log("worker.container_started", version=VERSION, python=sys.version.split()[0])
try:
    warm_up(settings_for_runpod())
except Exception as exc:  # noqa: BLE001 — le conteneur démarre quand même ; handle() rendra l'erreur à RunPod
    # Le message seulement pour une erreur de configuration : il est écrit par nous, sans secret.
    error("worker.container_setup_failed", error=type(exc).__name__, reason=str(exc)[:200] if isinstance(exc, ConfigError) else None)
    if not isinstance(exc, ConfigError):
        logger.warning("Pile d'appels du démarrage :\n%s", stack(exc))

runpod.serverless.start({"handler": handle})
