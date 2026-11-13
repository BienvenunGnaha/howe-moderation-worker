"""La boucle du travailleur (spécification, sections 8, 9 et 11).

- Il demande des tâches tant qu'il a des places libres (POST /tasks/lease, attente
  longue côté Howé).
- Chaque tâche tourne dans un dossier temporaire à elle, effacé à la fin, même en cas
  d'échec ; un battement de cœur prolonge le bail toutes les 20 s et arrête la tâche
  si Howé l'a annulée.
- Il ne reprend rien de lui-même après un redémarrage : Howé reprend ses tâches à
  l'expiration de leur bail.
- Arrêt propre (SIGTERM) : il ne prend plus rien et finit ses tâches en cours.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import measure, photo_check, video_frames_read, video_prepare, video_transcribe
from .client import HoweClient, HoweUnavailable, Refused
from .config import Settings
from .logs import log

Handler = Callable[..., dict[str, Any]]

HANDLERS: dict[str, Handler] = {
    "photo_check": photo_check.run,
    # Lot 4b : la vidéo.
    "video_prepare": video_prepare.run,
    "video_frames_read": video_frames_read.run,
    "video_transcribe": video_transcribe.run,
}

# Capacité de chaque type de tâche (pour compter les places libres).
CAPABILITY_OF = {"photo_check": "image", "video_prepare": "video", "video_frames_read": "image", "video_transcribe": "asr"}

# Les tâches qui déposent des fichiers chez Howé reçoivent la tâche elle-même (référence et bail).
NEEDS_TASK = {"video_prepare"}


class Runner:
    def __init__(self, settings: Settings, client: HoweClient, handlers: dict[str, Handler] | None = None):
        self.settings = settings
        self.client = client
        self.handlers = handlers or HANDLERS
        self.stopping = threading.Event()
        self._busy: dict[str, int] = {c: 0 for c in settings.capabilities}
        self._lock = threading.Lock()
        # Bilan des tâches traitées (rendu par le point d'entrée RunPod, lot 6) : des nombres seulement.
        self.outcomes: Counter[str] = Counter()
        settings.tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── Boucle ───────────────────────────────────────────────────────────────

    def run_forever(self) -> None:
        """La boucle. Sur RunPod (lot 6), elle s'arrête seule : plus de tâche depuis
        IDLE_EXIT_SECONDS, ou MAX_RUN_SECONDS écoulées. Elle finit toujours ses tâches en cours."""
        pause = 1.0
        started = time.monotonic()
        last_activity = started
        idle_exit = self.settings.idle_exit_seconds
        max_run = self.settings.max_run_seconds
        with ThreadPoolExecutor(max_workers=self.settings.total_slots) as pool:
            running: set[Future] = set()
            while not self.stopping.is_set():
                running = {f for f in running if not f.done()}
                now = time.monotonic()
                if running:
                    last_activity = now
                if max_run and now - started >= max_run:
                    log("worker.max_run_reached", running=len(running))
                    break
                if idle_exit and not running and now - last_activity >= idle_exit:
                    log("worker.idle_exit", idle_seconds=int(now - last_activity))
                    break
                free = self._free_capabilities()
                if not free:
                    time.sleep(0.5)
                    continue
                try:
                    tasks = self.client.lease(list(free), sum(free.values()), self.settings.lease_wait)
                    pause = 1.0
                except Refused as exc:
                    # Signature refusée, révoqué ou version trop ancienne : inutile d'insister vite.
                    log("worker.refused", status=exc.status, code=exc.code)
                    if idle_exit:
                        break  # sur RunPod, attendre coûterait du temps de GPU pour rien
                    self.stopping.wait(60)
                    continue
                except HoweUnavailable as exc:
                    log("worker.howe_unavailable", reason=str(exc))
                    self.stopping.wait(pause)
                    pause = min(pause * 2, 60.0)
                    continue
                if tasks:
                    last_activity = time.monotonic()
                for task in tasks:
                    self._reserve(task)
                    future = pool.submit(self.process, task)
                    future.add_done_callback(lambda _f, t=task: self._release(t))
                    running.add(future)
                running = {f for f in running if not f.done()}
            log("worker.draining", running=len(running))
        log("worker.stopped")

    def run_once(self) -> int:
        """Une seule prise de tâches, traitées à la suite (essais, mise en service)."""
        free = self._free_capabilities()
        tasks = self.client.lease(list(free), sum(free.values()), 0) if free else []
        for task in tasks:
            self.process(task)
        return len(tasks)

    # ── Une tâche ────────────────────────────────────────────────────────────

    def process(self, task: dict[str, Any]) -> str:
        reference, token, kind = task["reference"], task["lease_token"], task["kind"]
        cancelled = threading.Event()
        done = threading.Event()
        workdir = Path(tempfile.mkdtemp(prefix="task-", dir=self.settings.tmp_dir))
        heart = threading.Thread(target=self._heartbeat, args=(reference, token, cancelled, done), daemon=True)
        heart.start()
        started = time.monotonic()
        measured = measure.start()
        frames_read: int | None = None
        outcome = "succeeded"

        def metrics() -> dict[str, int]:
            # Rendues à Howé (0.3.0) : jamais de texte lu, seulement des nombres.
            values = {"duration_ms": int((time.monotonic() - started) * 1000), "worker_memory_mb": measure.worker_memory_mb()}
            if measured.tools_memory_kb:
                values["tools_memory_mb"] = max(1, measured.tools_memory_kb // 1024)
            if frames_read is not None:
                values["frames_read"] = frames_read
            return values

        try:
            handler = self.handlers.get(kind)
            if handler is None:
                self.client.fail(reference, token, "kind_unsupported", False, kind)
                outcome = "unsupported"
                return outcome
            extra = {"task": task} if kind in NEEDS_TASK else {}
            output = handler(task.get("input") or {}, workdir, self.client, self.settings, lambda: not cancelled.is_set(), **extra)
            read = (output.get("metrics") or {}).get("frames_read") if isinstance(output, dict) else None
            frames_read = read if isinstance(read, int) else None
            if cancelled.is_set():
                outcome = "cancelled"
            elif not self.client.complete(reference, token, output, metrics()):
                outcome = "stale"  # le bail n'était plus à nous : Howé ignore le résultat
        except photo_check.Cancelled:
            outcome = "cancelled"
        except photo_check.TaskFailed as exc:
            outcome = "failed"
            self._safe_fail(reference, token, exc.code, exc.retryable, exc.detail, metrics())
        except (HoweUnavailable, Refused) as exc:
            # Howé injoignable au moment de rendre : le bail expirera, la tâche reviendra dans la file.
            outcome = "unreported"
            log("task.unreported", task=reference, reason=str(exc))
        except Exception as exc:  # noqa: BLE001 — une tâche ne fait jamais tomber le travailleur
            outcome = "failed"
            self._safe_fail(reference, token, "worker_error", True, type(exc).__name__, metrics())
        finally:
            done.set()
            shutil.rmtree(workdir, ignore_errors=True)
            log("task.finished", task=reference, kind=kind, outcome=outcome, **metrics())
            with self._lock:
                self.outcomes[outcome] += 1
        return outcome

    def _heartbeat(self, reference: str, token: str, cancelled: threading.Event, done: threading.Event) -> None:
        while not done.wait(self.settings.heartbeat_seconds):
            try:
                if not self.client.heartbeat(reference, token):
                    cancelled.set()
                    return
            except (HoweUnavailable, Refused):
                continue  # le prochain battement réessaiera ; au pire, le bail expire

    def _safe_fail(self, reference: str, token: str, code: str, retryable: bool, detail: str | None,
                   metrics: dict[str, int] | None = None) -> None:
        try:
            self.client.fail(reference, token, code, retryable, detail, metrics)
        except (HoweUnavailable, Refused) as exc:
            log("task.unreported", task=reference, reason=str(exc))

    # ── Places ───────────────────────────────────────────────────────────────

    def _free_capabilities(self) -> dict[str, int]:
        with self._lock:
            return {c: self.settings.slots.get(c, 0) - self._busy.get(c, 0)
                    for c in self.settings.capabilities if self.settings.slots.get(c, 0) - self._busy.get(c, 0) > 0}

    def _reserve(self, task: dict[str, Any]) -> None:
        with self._lock:
            capability = CAPABILITY_OF.get(task["kind"], "image")
            self._busy[capability] = self._busy.get(capability, 0) + 1

    def _release(self, task: dict[str, Any]) -> None:
        with self._lock:
            capability = CAPABILITY_OF.get(task["kind"], "image")
            self._busy[capability] = max(0, self._busy.get(capability, 0) - 1)
