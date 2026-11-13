"""La signature des appels : le message doit être exactement celui que Howé vérifie
(RelayClient::message), et la clé privée ne sort jamais du travailleur."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from pathlib import Path

import httpx
import pytest
from nacl.signing import SigningKey

from howe_worker import VERSION
from howe_worker.client import HoweClient, Refused, message
from howe_worker.config import ConfigError, Settings, load


def test_message_is_the_relay_one() -> None:
    body = b'{"slots":1}'
    expected = "POST\n/api/moderation/v1/tasks/lease\n1700000000\nabcd\n\n" + hashlib.sha256(body).hexdigest()
    assert message("post", "/api/moderation/v1/tasks/lease", "1700000000", "abcd", body) == expected.encode()


def test_calls_are_signed_on_the_full_path(settings: Settings) -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(204)

    prefixed = Settings(**{**settings.__dict__, "howe_url": "https://howe.test/sous-dossier"})
    client = HoweClient(prefixed, httpx.Client(base_url=prefixed.howe_url, transport=httpx.MockTransport(handler)))
    assert client.lease(["image"], 2, 0) == []

    request = seen["request"]
    headers = request.headers
    assert headers["X-Worker"] == prefixed.reference
    assert request.url.path == "/sous-dossier/api/moderation/v1/tasks/lease"
    signed = message("POST", request.url.path, headers["X-Timestamp"], headers["X-Nonce"], request.content)
    SigningKey(prefixed.seed).verify_key.verify(signed, base64.b64decode(headers["X-Signature"]))
    assert json.loads(request.content) == {"capabilities": ["image"], "slots": 2, "version": VERSION, "wait": 0}
    assert base64.b64encode(prefixed.seed).decode() not in str(dict(headers))


def test_a_refused_worker_is_told_why(settings: Settings) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(409, json={"code": "worker_outdated"}))
    client = HoweClient(settings, httpx.Client(base_url=settings.howe_url, transport=transport))
    with pytest.raises(Refused) as caught:
        client.lease(["image"], 1, 0)
    assert caught.value.code == "worker_outdated"


def test_a_stale_lease_is_not_an_error(settings: Settings) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(409, json={"detail": "plus à vous"}))
    client = HoweClient(settings, httpx.Client(base_url=settings.howe_url, transport=transport))
    assert client.complete("t-1", "jeton", {"text": []}) is False


def test_the_key_file_must_be_private(tmp_path: Path) -> None:
    key = tmp_path / "worker.key"
    key.write_text(base64.b64encode(os.urandom(32)).decode())
    env = {"HOWE_URL": "https://howe.test", "WORKER_REFERENCE": "0" * 8 + "-0000-4000-8000-" + "0" * 12,
           "WORKER_PRIVATE_KEY_FILE": str(key), "SLOTS": "image=3"}

    key.chmod(0o644)
    with pytest.raises(ConfigError, match="0600"):
        load(env)

    key.chmod(stat.S_IRUSR | stat.S_IWUSR)
    settings = load(env)
    assert settings.slots == {"image": 3} and len(settings.seed) == 32
    assert "seed" not in repr(settings) or base64.b64encode(settings.seed).decode() not in repr(settings)
