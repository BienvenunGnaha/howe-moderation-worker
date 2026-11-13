"""Point d'entrée RunPod Serverless. La logique est dans howe_worker/serverless.py (README, lot 6)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runpod  # noqa: E402

from howe_worker.serverless import handle, quiet_http_logs, settings_for_runpod, warm_up  # noqa: E402

quiet_http_logs()
try:
    warm_up(settings_for_runpod())
except Exception:  # noqa: BLE001 — la configuration sera rapportée par handle()
    pass

runpod.serverless.start({"handler": handle})
