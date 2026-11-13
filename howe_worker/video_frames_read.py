"""video_frames_read : le texte et les QR codes des images d'une vidéo (spécification,
section 3.3 ; lot 4b).

Les images ont été extraites par video_prepare (deux par seconde, plus chaque
changement de plan, doublons écartés) ; chacune est lue comme une photo, par le
lien signé que donne la tâche, avec son instant. Une lecture Tesseract par image
(pas deux, comme pour une photo) : une vidéo en donne jusqu'à 120, et le délai
de la tâche doit tenir.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .client import DownloadRefused, HoweClient
from .config import Settings
from .logs import step
from .photo_check import Cancelled, TaskFailed, open_checked, read_qr, tesseract_lines, tesseract_version


def run(task_input: dict[str, Any], workdir: Path, client: HoweClient, settings: Settings,
        should_continue: Callable[[], bool]) -> dict[str, Any]:
    started = time.monotonic()
    frames = task_input.get("frames") or []
    limits = task_input.get("limits") or {}
    budget = float(task_input.get("timeout") or 300) - 20
    per_frame = max(5.0, min(30.0, budget / max(len(frames), 1)))
    results: list[dict[str, Any]] = []
    engines: set[str] = set()

    for index, frame in enumerate(frames):
        if not should_continue():
            raise Cancelled()
        path = workdir / f"frame{index:04d}.jpg"
        try:
            client.download(frame["url"], path, 2 * 1024 * 1024)
        except DownloadRefused as exc:
            raise TaskFailed(exc.code, exc.retryable) from exc
        # Une ligne par image en debug : le nombre de lignes lues, jamais leur texte.
        with step("frame", index=index, at_ms=int(frame.get("at_ms") or 0)) as noted:
            image = open_checked(path, settings, limits)
            plain = workdir / f"frame{index:04d}.png"
            image.save(plain)
            lines = tesseract_lines(plain, "frame", settings, should_continue, per_frame, psm=3)
            qr, qr_engines = read_qr([image])
            noted.update(lines=len(lines), codes=len(qr))
        engines.update(qr_engines)
        results.append({"at_ms": int(frame.get("at_ms") or 0), "text": lines, "qr": qr})
        path.unlink(missing_ok=True)
        plain.unlink(missing_ok=True)

    return {
        "frames": results,
        "engines": [tesseract_version(settings), *sorted(engines)],
        "metrics": {"duration_ms": int((time.monotonic() - started) * 1000), "frames_read": len(results)},
    }
