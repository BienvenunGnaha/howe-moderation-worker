"""Le journal, comme celui d'ocr-worker (13/11/2026) : `logging`, des phrases lisibles, un
chronomètre par tâche avec son résumé. Assez de détails pour trouver une panne, jamais de
texte lu, de lien signé, de bail ni de clé."""

from __future__ import annotations

import logging
import re
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


@pytest.fixture(autouse=True)
def info_logs(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    caplog.set_level(logging.INFO, logger="howe_worker")
    yield
    logs.set_level("info")


def messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "howe_worker"]


def test_the_format_is_the_one_of_ocr_worker() -> None:
    root = logging.getLogger()
    formats = {getattr(h.formatter, "_fmt", None) for h in root.handlers}
    # basicConfig est sans effet si un autre module l'a appelé avant (pytest) ; le format, lui, est écrit dans logs.py.
    source = Path(logs.__file__).read_text()
    assert 'format="%(asctime)s %(levelname)-8s %(name)s — %(message)s"' in source
    assert logs.logger.name == "howe_worker" and formats is not None


def test_each_step_of_a_task_is_timed_then_summed_up(settings: Settings, tmp_path: Path, caplog) -> None:
    howe = FakeHowe({"photo": sign_photo(tmp_path / "panneau.png", "A LOUER 97 45 32 21")})

    Runner(settings, howe).process(task("4c4c61b6-1942-4403-8c8e-c14b215a6ea3"))  # type: ignore[arg-type]

    lines = messages(caplog)
    assert lines[0].startswith("Tâche 4c4c61b6 prise : photo_check")
    chrono = [l for l in lines if l.startswith("[CHRONO] 4c4c61b6 · ")]
    steps = [re.match(r"\[CHRONO\] 4c4c61b6 · (.+?) — ", l).group(1) for l in chrono[:-1]]
    assert steps == ["téléchargement", "ouverture", "lecture Tesseract (original)", "lecture Tesseract (contrast)", "QR codes", "rendu à Howé"]
    assert re.search(r"— \d+,\d\d s \(cumul \d+,\d\d s\)", chrono[0])
    # Le résumé, trié par durée, avec l'issue de la tâche.
    assert re.match(r"\[CHRONO\] 4c4c61b6 · TOTAL \d+,\d\d s — .+ \(\d+ %\).* — photo_check, succeeded$", chrono[-1])
    assert "Tâche 4c4c61b6 terminée : photo_check, succeeded" in "\n".join(lines)
    # Ni le texte lu, ni le lien du fichier.
    assert "LOUER" not in caplog.text and "97 45" not in caplog.text


def test_at_the_warning_level_only_the_problems_are_written(settings: Settings, tmp_path: Path, caplog) -> None:
    logs.set_level("warning")
    howe = FakeHowe({"photo": sign_photo(tmp_path / "panneau.png", "A LOUER")})

    Runner(settings, howe).process(task("t-1"))  # type: ignore[arg-type]
    Runner(settings, FakeHowe()).process(task("t-2", url="absente"))  # type: ignore[arg-type]

    lines = messages(caplog)
    assert lines and all("t-2" in l or l.startswith("Pile d'appels") for l in lines)
    assert {r.levelname for r in caplog.records if r.name == "howe_worker"} == {"ERROR", "WARNING"}


def test_a_failure_says_its_code_and_a_crash_shows_its_stack_but_never_its_message(settings: Settings, caplog) -> None:
    # Le message est fabriqué à l'exécution : la pile montre les lignes de code, jamais les valeurs.
    secret_message = " ".join(["échec", "de", "lecture", SIGNED])

    def broken(*_args):
        raise ValueError(secret_message)

    Runner(settings, FakeHowe(), {"photo_check": broken}).process(task("t-7"))  # type: ignore[arg-type]

    assert "signature=" not in caplog.text and "échec de lecture" not in caplog.text
    lines = messages(caplog)
    assert any(l.startswith("Tâche t-7 (photo_check) : erreur inattendue ValueError") for l in lines)
    stack = next(l for l in lines if l.startswith("Pile d'appels de la tâche t-7"))
    assert "test_logs.py" in stack and "in broken" in stack and stack.endswith("ValueError")
    finished = next(l for l in lines if l.startswith("Tâche t-7 terminée"))
    assert "failed" in finished and "code=worker_error" in finished
    assert any(l.startswith("[CHRONO] t-7 · TOTAL") and l.endswith("photo_check, failed") for l in lines)


def test_the_cancellation_by_howe_is_written(settings: Settings, caplog) -> None:
    import time

    def slow(_input, _workdir, _client, _settings, should_continue):
        while should_continue():
            time.sleep(0.05)
        return {}

    Runner(settings, FakeHowe(keep_going=False), {"photo_check": slow}).process(task("t-3"))  # type: ignore[arg-type]

    lines = messages(caplog)
    assert "Tâche t-3 annulée par Howé : arrêt" in lines
    assert any(l.startswith("Tâche t-3 terminée : photo_check, cancelled") for l in lines)


def test_calls_and_downloads_are_logged_without_the_link_or_the_lease(settings: Settings, tmp_path: Path, caplog) -> None:
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

    assert "signature=" not in caplog.text and "expires=" not in caplog.text and "jeton-secret" not in caplog.text
    download, call = messages(caplog)
    assert re.match(r"^Fichier original lu : 2 048 octets en \d+ ms$", download)
    assert re.match(r"^Howé POST /tasks/t-9/heartbeat → 200 en \d+ ms$", call)


def test_a_refused_worker_is_told_what_to_check(settings: Settings, caplog) -> None:
    class Revoked(FakeHowe):
        def lease(self, *_args):
            raise Refused(401, "signature")

    from dataclasses import replace

    Runner(replace(settings, idle_exit_seconds=5), Revoked()).run_forever()  # type: ignore[arg-type]

    refused = next(r for r in caplog.records if r.getMessage().startswith("Howé refuse le travailleur"))
    assert refused.levelname == "ERROR" and "WORKER_REFERENCE" in refused.getMessage()


def test_the_start_line_names_the_host_but_never_the_key(settings: Settings) -> None:
    from howe_worker.runner import startup_fields

    fields = startup_fields(settings)

    assert fields["howe"] == "http://howe.test" and fields["worker"] == settings.reference[:8]
    assert settings.seed.hex() not in logs.render("worker.started", fields)
    assert logs.host_of(SIGNED) == "https://howe.test"
