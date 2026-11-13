"""Le journal (spécification, section 16) : assez de détails pour suivre une tâche et
trouver une panne, mais jamais de texte lu, de lien signé, de bail ni de clé."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from howe_worker import logs
from howe_worker.client import HoweClient, Refused
from howe_worker.config import Settings
from howe_worker.runner import Runner

from .conftest import FakeHowe, sign_photo, task

SIGNED = "https://howe.test/api/moderation/v1/files/abc?expires=1700000000&signature=0123456789abcdef"


@pytest.fixture
def warning_level() -> Iterator[None]:
    logs.set_level("warning")
    yield
    logs.set_level("info")


def lines(out: str) -> list[dict]:
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def test_by_default_each_step_of_a_task_is_timed_and_tied_to_the_task(settings: Settings, tmp_path: Path, capsys) -> None:
    howe = FakeHowe({"photo": sign_photo(tmp_path / "panneau.png", "A LOUER 97 45 32 21")})

    Runner(settings, howe).process(task("t-42"))  # type: ignore[arg-type]

    records = lines(capsys.readouterr().out)
    events = [r["event"] for r in records]
    assert events[0] == "task.started" and events[-1] == "task.finished"
    steps = [(r["step"], r.get("variant")) for r in records if r["event"] == "step.done"]
    assert steps == [("open", None), ("tesseract", "original"), ("tesseract", "contrast"), ("qr", None)]
    # Chaque ligne écrite pendant la tâche porte sa référence et son type, sans qu'on les passe à chaque appel.
    assert all(r["task"] == "t-42" and r["kind"] == "photo_check" for r in records)
    assert all(isinstance(r["ms"], int) for r in records if r["event"] == "step.done")
    tesseract = next(r for r in records if r.get("step") == "tesseract")
    assert tesseract["lines"] > 0 and tesseract["level"] == "info"


def test_at_the_warning_level_only_the_problems_are_written(settings: Settings, tmp_path: Path, capsys, warning_level) -> None:
    howe = FakeHowe({"photo": sign_photo(tmp_path / "panneau.png", "A LOUER")})

    Runner(settings, howe).process(task("t-1"))  # type: ignore[arg-type]
    Runner(settings, FakeHowe()).process(task("t-2", url="absente"))  # type: ignore[arg-type]

    records = lines(capsys.readouterr().out)
    assert [(r["event"], r["task"]) for r in records] == [("task.crashed", "t-2"), ("task.finished", "t-2")]
    assert {r["level"] for r in records} == {"error", "warning"}


def test_a_failure_says_its_code_and_a_crash_says_where_but_never_its_message(settings: Settings, capsys) -> None:
    def broken(*_args):
        raise ValueError(f"lecture impossible : {SIGNED}")

    Runner(settings, FakeHowe(), {"photo_check": broken}).process(task("t-7"))  # type: ignore[arg-type]
    Runner(settings, FakeHowe()).process(task("t-8", url="absente"))  # type: ignore[arg-type]

    out = capsys.readouterr().out
    assert "signature=" not in out and "lecture impossible" not in out
    records = lines(out)
    crash = next(r for r in records if r["event"] == "task.crashed")
    assert crash["error"] == "ValueError" and crash["where"][0].startswith("test_logs.py:")
    finished = [r for r in records if r["event"] == "task.finished"]
    assert [(r["task"], r["outcome"], r["code"], r["level"]) for r in finished] == [
        ("t-7", "failed", "worker_error", "warning"),
        ("t-8", "failed", "worker_error", "warning"),
    ]


def test_the_cancellation_by_howe_is_written(settings: Settings, tmp_path: Path, capsys) -> None:
    import time

    def slow(_input, _workdir, _client, _settings, should_continue):
        while should_continue():
            time.sleep(0.05)
        return {}

    Runner(settings, FakeHowe(keep_going=False), {"photo_check": slow}).process(task("t-3"))  # type: ignore[arg-type]

    records = lines(capsys.readouterr().out)
    assert any(r["event"] == "task.cancelled_by_howe" and r["task"] == "t-3" for r in records)
    assert records[-1]["outcome"] == "cancelled"


def test_calls_and_downloads_are_logged_without_the_link_or_the_lease(settings: Settings, tmp_path: Path, capsys) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/heartbeat"):
            return httpx.Response(200, json={"continue": True})
        if request.url.path.endswith("/tasks/lease"):
            return httpx.Response(204)
        return httpx.Response(200, content=b"x" * 2048)

    client = HoweClient(settings, httpx.Client(base_url=settings.howe_url, transport=httpx.MockTransport(handler)))
    client.download(SIGNED, tmp_path / "original", 10_000)
    client.heartbeat("t-9", "jeton-secret-du-bail")
    # Une demande restée vide n'est pas écrite : sans travail, elle revient toutes les 20 s.
    assert client.lease(["image"], 1, 0) == []

    out = capsys.readouterr().out
    assert "signature=" not in out and "expires=" not in out and "jeton-secret" not in out
    download, call = lines(out)
    assert (download["event"], download["name"], download["bytes"]) == ("file.download", "original", 2048)
    assert (call["event"], call["method"], call["path"], call["status"]) == ("howe.call", "POST", "/tasks/t-9/heartbeat", 200)


def test_a_refused_worker_is_told_what_to_check(settings: Settings, capsys) -> None:
    class Revoked(FakeHowe):
        def lease(self, *_args):
            raise Refused(401, "signature")

    from dataclasses import replace

    Runner(replace(settings, idle_exit_seconds=5), Revoked()).run_forever()  # type: ignore[arg-type]

    refused = next(r for r in lines(capsys.readouterr().out) if r["event"] == "worker.refused")
    assert refused["level"] == "error" and "WORKER_REFERENCE" in refused["hint"]


def test_the_start_line_names_the_host_but_never_the_key(settings: Settings) -> None:
    from howe_worker.runner import startup_fields

    fields = startup_fields(settings)

    assert fields["howe"] == "http://howe.test" and fields["worker"] == settings.reference[:8]
    assert settings.seed.hex() not in json.dumps(fields, default=str)
    assert logs.host_of(SIGNED) == "https://howe.test"
