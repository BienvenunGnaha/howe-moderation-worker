"""Mesures d'une tâche, rendues à Howé avec son résultat (version 0.3.0, spécification,
section 16) : elles nourrissent l'écran « Modération automatique » et le dimensionnement
des serveurs, sans que personne n'ait à se connecter sur la machine.

- tools_memory_mb : la mémoire maximale d'un programme lancé pour la tâche (Tesseract,
  FFmpeg, FFprobe), lue exactement à sa fin par wait4 ;
- worker_memory_mb : le plus haut niveau de mémoire du travailleur depuis son démarrage
  (Whisper, chargé une fois, y compris). Le système ne la donne pas tâche par tâche.

Plusieurs tâches tournent en même temps, chacune dans son fil : chaque fil a son relevé.
"""

from __future__ import annotations

import os
import resource
import subprocess
import threading
from dataclasses import dataclass

_local = threading.local()


@dataclass
class TaskMeasure:
    tools_memory_kb: int = 0


def start() -> TaskMeasure:
    """Un relevé neuf pour la tâche qui commence dans ce fil."""
    _local.measure = TaskMeasure()
    return _local.measure


def current() -> TaskMeasure | None:
    return getattr(_local, "measure", None)


def note_tool(max_rss_kb: int) -> None:
    measure = current()
    if measure is not None and max_rss_kb > measure.tools_memory_kb:
        measure.tools_memory_kb = max_rss_kb


def worker_memory_mb() -> int:
    # ru_maxrss est en kilo-octets sous Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


class MeasuredPopen(subprocess.Popen):
    """Popen qui garde la mémoire maximale du programme à sa fin (wait4 au lieu de waitpid).

    Remplace une méthode interne de Popen, stable de Python 3.8 à 3.13 : si elle venait
    à changer, la mesure manquerait, mais le programme tournerait comme avant.
    """

    max_rss_kb = 0

    def _try_wait(self, wait_flags):  # noqa: ANN001 — même signature que Popen
        try:
            pid, status, usage = os.wait4(self.pid, wait_flags)
        except ChildProcessError:
            return self.pid, 0
        if pid == self.pid:
            self.max_rss_kb = usage.ru_maxrss
        return pid, status
