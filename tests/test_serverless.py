"""RunPod Serverless (lot 6) : un réveil traite ce qui attend, puis s'arrête seul.

Sans le paquet runpod, sans GPU et sans réseau : Howé est simulé.
"""

from __future__ import annotations

import base64
import dataclasses
import os
import time
from pathlib import Path

import pytest

from howe_worker.client import Refused
from howe_worker.config import ConfigError, Settings
from howe_worker.serverless import handle, settings_for_runpod

from .conftest import FakeHowe, sign_photo, task


def runpod_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {
        "HOWE_URL": "https://howe.test",
        "WORKER_REFERENCE": "00000000-0000-4000-8000-000000000001",
        # Sur RunPod, la clé vient d'un secret de l'endpoint, jamais d'un fichier de l'image.
        "WORKER_PRIVATE_KEY": base64.b64encode(os.urandom(32)).decode(),
        "TMP_DIR": str(tmp_path / "work"),
        **extra,
    }


def test_runpod_settings_stop_by_themselves_and_read_the_secret_key(tmp_path: Path) -> None:
    settings = settings_for_runpod(runpod_env(tmp_path))

    assert (settings.idle_exit_seconds, settings.max_run_seconds) == (20, 1800)
    assert len(settings.seed) == 32
    with pytest.raises(ConfigError):
        settings_for_runpod(runpod_env(tmp_path, IDLE_EXIT_SECONDS="0"))


def test_whisper_runs_on_the_gpu_when_asked(tmp_path: Path) -> None:
    settings = settings_for_runpod(runpod_env(tmp_path, WHISPER_DEVICE="cuda", WHISPER_COMPUTE_TYPE="float16"))

    assert (settings.whisper_device, settings.whisper_compute_type) == ("cuda", "float16")
    with pytest.raises(ConfigError):
        settings_for_runpod(runpod_env(tmp_path, WHISPER_DEVICE="tpu"))


def test_a_wake_up_takes_the_waiting_tasks_then_stops_when_idle(settings: Settings, tmp_path: Path) -> None:
    howe = FakeHowe({"photo": sign_photo(tmp_path / "panneau.png", "A LOUER")})
    howe.leases = [[task("t-1")], [task("t-2")]]
    quick = dataclasses.replace(settings, idle_exit_seconds=1, max_run_seconds=60)

    started = time.monotonic()
    result = handle({"id": "rp-1", "input": {}}, quick, howe)  # type: ignore[arg-type]

    assert result == {"status": "ok", "version": result["version"], "outcomes": {"succeeded": 2}}
    # Traitées en parallèle : l'ordre de fin ne compte pas.
    assert sorted(r for r, _ in howe.completed) == ["t-1", "t-2"]
    assert time.monotonic() - started < 15, "il s'arrête peu après la dernière tâche"
    assert list(quick.tmp_dir.iterdir()) == []


def test_a_wake_up_stops_at_once_when_howe_refuses_it(settings: Settings) -> None:
    class Revoked(FakeHowe):
        def lease(self, *_args):
            raise Refused(401, "worker_revoked")

    quick = dataclasses.replace(settings, idle_exit_seconds=20, max_run_seconds=60)
    started = time.monotonic()

    result = handle({"id": "rp-2"}, quick, Revoked())  # type: ignore[arg-type]

    assert result["outcomes"] == {}
    assert time.monotonic() - started < 5, "révoqué : pas une minute de GPU pour rien"


def test_a_wake_up_never_outlives_its_maximum(settings: Settings) -> None:
    class Endless(FakeHowe):
        def lease(self, *_args):
            time.sleep(0.2)
            return []

    # Même sans jamais être inactif assez longtemps, la durée maximale l'arrête.
    quick = dataclasses.replace(settings, idle_exit_seconds=60, max_run_seconds=1)
    started = time.monotonic()

    assert handle({"id": "rp-3"}, quick, Endless())["status"] == "ok"  # type: ignore[arg-type]
    assert time.monotonic() - started < 5


def test_signed_links_never_reach_the_logs(settings: Settings, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    # Comme le fait le SDK RunPod : tout le monde en INFO.
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.INFO)
    quick = dataclasses.replace(settings, idle_exit_seconds=1, max_run_seconds=10)

    handle({"id": "rp-5"}, quick, FakeHowe())  # type: ignore[arg-type]
    logging.getLogger("httpx").info("HTTP Request: GET https://howe.test/api/moderation/v1/files/x?signature=secret")

    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING
    assert "signature=" not in caplog.text


def test_a_bad_configuration_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOWE_URL", raising=False)
    monkeypatch.delenv("WORKER_REFERENCE", raising=False)

    result = handle({"id": "rp-4"})

    assert result["status"] == "error" and result["code"] == "config"
