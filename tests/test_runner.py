"""La boucle : résultat rendu, échec rendu, arrêt sur annulation, dossier temporaire
toujours effacé, et le texte lu jamais écrit dans le journal."""

from __future__ import annotations

import time
from pathlib import Path

from howe_worker.config import Settings
from howe_worker.runner import Runner

from .conftest import FakeHowe, sign_photo, task


def test_a_result_is_returned_and_the_workdir_removed(settings: Settings, tmp_path: Path, caplog) -> None:
    caplog.set_level("INFO", logger="howe_worker")
    photo = sign_photo(tmp_path / "panneau.png", "A LOUER 97 45 32 21")
    howe = FakeHowe({"photo": photo})

    outcome = Runner(settings, howe).process(task())  # type: ignore[arg-type]

    assert outcome == "succeeded"
    [(reference, output)] = howe.completed
    assert reference == "t-1" and output["text"]
    assert list(settings.tmp_dir.iterdir()) == []
    # Le journal ne porte ni le texte lu ni le lien du fichier.
    logged = caplog.text
    assert "Tâche t-1 terminée" in logged
    assert "LOUER" not in logged and "97 45" not in logged and "photo" not in logged.replace("photo_check", "")


def test_the_task_measures_go_back_with_the_result(settings: Settings, tmp_path: Path, caplog) -> None:
    caplog.set_level("INFO", logger="howe_worker")
    # 0.3.0 (lot 5) : durée, mémoire de Tesseract (lue à sa fin) et du travailleur, rien d'autre.
    howe = FakeHowe({"photo": sign_photo(tmp_path / "panneau.png", "A LOUER")})

    Runner(settings, howe).process(task())  # type: ignore[arg-type]

    [(reference, metrics)] = howe.metrics
    assert reference == "t-1"
    assert set(metrics) == {"duration_ms", "worker_memory_mb", "tools_memory_mb"}
    assert metrics["tools_memory_mb"] >= 1 and metrics["worker_memory_mb"] >= 1 and metrics["duration_ms"] > 0
    assert "tools_memory_mb=" in caplog.text


def test_the_frames_read_are_measured_and_a_failure_carries_its_measures(settings: Settings) -> None:
    def frames(*_args):
        return {"frames": [], "metrics": {"duration_ms": 5, "frames_read": 42}}

    def broken(*_args):
        raise RuntimeError("panne")

    howe = FakeHowe()
    Runner(settings, howe, {"photo_check": frames}).process(task())  # type: ignore[arg-type]
    Runner(settings, howe, {"photo_check": broken}).process(task("t-2"))  # type: ignore[arg-type]

    (_, first), (_, second) = howe.metrics
    assert first["frames_read"] == 42 and "tools_memory_mb" not in first
    assert set(second) == {"duration_ms", "worker_memory_mb"}


def test_two_tasks_side_by_side_keep_their_own_measures(settings: Settings, tmp_path: Path) -> None:
    import threading

    from howe_worker import measure

    seen: dict[str, int] = {}

    def heavy(name: str, kb: int):
        measure.start()
        measure.note_tool(kb)
        time.sleep(0.05)
        seen[name] = measure.current().tools_memory_kb

    threads = [threading.Thread(target=heavy, args=("a", 300_000)), threading.Thread(target=heavy, args=("b", 20_000))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen == {"a": 300_000, "b": 20_000}


def test_a_refused_file_is_reported_as_such(settings: Settings, tmp_path: Path) -> None:
    fake = tmp_path / "faux.jpg"
    fake.write_bytes(b"pas une image")
    howe = FakeHowe({"photo": fake})

    assert Runner(settings, howe).process(task()) == "failed"  # type: ignore[arg-type]
    assert howe.failed == [("t-1", "file_unsupported", False)]
    assert list(settings.tmp_dir.iterdir()) == []


def test_an_unexpected_error_is_retryable_and_never_kills_the_worker(settings: Settings) -> None:
    def broken(*_args):
        raise RuntimeError("panne")

    howe = FakeHowe()
    assert Runner(settings, howe, {"photo_check": broken}).process(task()) == "failed"  # type: ignore[arg-type]
    assert howe.failed == [("t-1", "worker_error", True)]


def test_a_task_cancelled_by_howe_is_not_returned(settings: Settings) -> None:
    from howe_worker.photo_check import Cancelled

    def cancellable(_input, _workdir, _client, _settings, keep_going):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not keep_going():
                raise Cancelled()
            time.sleep(0.05)
        return {"text": []}

    howe = FakeHowe(keep_going=False)
    outcome = Runner(settings, howe, {"photo_check": cancellable}).process(task())  # type: ignore[arg-type]

    assert outcome == "cancelled"
    assert howe.heartbeats >= 1 and howe.completed == []
    assert list(settings.tmp_dir.iterdir()) == []


def test_run_once_takes_what_it_has_room_for(settings: Settings, tmp_path: Path) -> None:
    photo = sign_photo(tmp_path / "panneau.png", "A LOUER")
    howe = FakeHowe({"photo": photo})
    howe.leases = [[task("t-1"), task("t-2")]]

    assert Runner(settings, howe).run_once() == 2  # type: ignore[arg-type]
    assert [r for r, _ in howe.completed] == ["t-1", "t-2"]
