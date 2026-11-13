"""Les appels à Howé, signés en Ed25519 (spécification, sections 8 et 13).

Le message signé est celui du Relais, que Howé vérifie déjà :
méthode, chemin, horodatage, nonce, ligne vide, empreinte SHA-256 du corps.
En-têtes : X-Worker, X-Timestamp, X-Nonce, X-Signature (base64).

Les fichiers se lisent par le lien signé que donne chaque tâche (15 minutes),
sans en-tête de signature ni compte.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from nacl.signing import SigningKey

from . import VERSION
from .config import Settings

API_PREFIX = "/api/moderation/v1"


class HoweUnavailable(Exception):
    """Howé injoignable ou en erreur : le travailleur attend et réessaie."""


class Refused(Exception):
    """Howé refuse le travailleur (signature, révocation, version trop ancienne)."""

    def __init__(self, status: int, code: str | None = None):
        super().__init__(f"refusé ({status}{', ' + code if code else ''})")
        self.status = status
        self.code = code


class LeaseLost(Exception):
    """Howé refuse un fichier déposé : le bail n'est plus à nous, ou le nom n'est pas attendu."""


class DownloadRefused(Exception):
    """Le fichier n'a pas pu être lu : trop gros, lien expiré, tâche plus à nous."""

    def __init__(self, code: str, retryable: bool):
        super().__init__(code)
        self.code = code
        self.retryable = retryable


def message(method: str, path: str, timestamp: str, nonce: str, body: bytes) -> bytes:
    """Le message signé, identique à RelayClient::message côté Howé."""
    return "\n".join([method.upper(), path, timestamp, nonce, "", hashlib.sha256(body).hexdigest()]).encode()


class HoweClient:
    def __init__(self, settings: Settings, http: httpx.Client | None = None):
        self.settings = settings
        self._key = SigningKey(settings.seed)
        self._prefix = urlsplit(settings.howe_url).path.rstrip("/")
        self._http = http or httpx.Client(
            base_url=settings.howe_url,
            timeout=httpx.Timeout(15.0, read=settings.lease_wait + 15.0),
            headers={"User-Agent": f"howe-moderation-worker/{VERSION}"},
        )

    # ── Appels signés ────────────────────────────────────────────────────────

    def signed_headers(self, method: str, path: str, body: bytes) -> dict[str, str]:
        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(16)
        signature = self._key.sign(message(method, self._prefix + path, timestamp, nonce, body)).signature
        return {
            "X-Worker": self.settings.reference,
            "X-Timestamp": timestamp,
            "X-Nonce": nonce,
            "X-Signature": base64.b64encode(signature).decode(),
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _call(self, method: str, path: str, payload: dict[str, Any] | None = None) -> httpx.Response:
        body = b"" if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        full = API_PREFIX + path
        try:
            response = self._http.request(method, full, content=body, headers=self.signed_headers(method, full, body))
        except httpx.HTTPError as exc:
            raise HoweUnavailable(type(exc).__name__) from exc
        if response.status_code == 401:
            raise Refused(401, "signature")
        if response.status_code >= 500:
            raise HoweUnavailable(f"http_{response.status_code}")
        return response

    def me(self) -> dict[str, Any]:
        response = self._call("GET", "/workers/me")
        response.raise_for_status()
        return response.json()

    def lease(self, capabilities: list[str], slots: int, wait: int) -> list[dict[str, Any]]:
        response = self._call("POST", "/tasks/lease", {"capabilities": capabilities, "slots": slots, "version": VERSION, "wait": wait})
        if response.status_code == 204:
            return []
        if response.status_code == 409:
            raise Refused(409, (response.json() or {}).get("code"))
        response.raise_for_status()
        return response.json().get("tasks", [])

    def heartbeat(self, reference: str, token: str, progress: int | None = None) -> bool:
        payload: dict[str, Any] = {"lease_token": token}
        if progress is not None:
            payload["progress"] = progress
        response = self._call("POST", f"/tasks/{reference}/heartbeat", payload)
        response.raise_for_status()
        return bool(response.json().get("continue"))

    def complete(self, reference: str, token: str, output: dict[str, Any], metrics: dict[str, int] | None = None) -> bool:
        """Faux si le bail n'est plus à nous (409) : le résultat est ignoré par Howé.

        metrics (0.3.0) : durée, mémoire, images lues ; Howé les garde avec la tâche.
        """
        payload: dict[str, Any] = {"lease_token": token, "output": output}
        if metrics:
            payload["metrics"] = metrics
        response = self._call("POST", f"/tasks/{reference}/complete", payload)
        if response.status_code == 409:
            return False
        response.raise_for_status()
        return True

    def fail(self, reference: str, token: str, code: str, retryable: bool, detail: str | None = None,
             metrics: dict[str, int] | None = None) -> None:
        payload: dict[str, Any] = {"lease_token": token, "error_code": code, "retryable": retryable}
        if detail:
            payload["detail"] = detail[:200]
        if metrics:
            payload["metrics"] = metrics
        self._call("POST", f"/tasks/{reference}/fail", payload).raise_for_status()

    def upload(self, reference: str, token: str, name: str, path: Path) -> None:
        """Dépose un fichier produit par la tâche (lot 4b) : appel signé, corps brut, bail en en-tête.

        Un 409 veut dire que le bail n'est plus à nous (tâche annulée ou reprise) : LeaseLost.
        """
        body = path.read_bytes()
        full = f"{API_PREFIX}/tasks/{reference}/files/{name}"
        headers = self.signed_headers("PUT", full, body)
        headers.update({"Content-Type": "application/octet-stream", "X-Lease-Token": token})
        try:
            response = self._http.put(full, content=body, headers=headers, timeout=httpx.Timeout(120.0))
        except httpx.HTTPError as exc:
            raise HoweUnavailable(type(exc).__name__) from exc
        if response.status_code == 401:
            raise Refused(401, "signature")
        if response.status_code in (400, 409):
            raise LeaseLost(name)
        if response.status_code >= 500:
            raise HoweUnavailable(f"http_{response.status_code}")
        response.raise_for_status()

    # ── Fichiers ─────────────────────────────────────────────────────────────

    def download(self, url: str, destination: Path, max_bytes: int) -> int:
        """Télécharge le fichier d'une tâche, sans jamais dépasser max_bytes sur le disque."""
        written = 0
        try:
            with self._http.stream("GET", url, timeout=httpx.Timeout(30.0)) as response:
                if response.status_code in (403, 404):
                    # Lien expiré, ou tâche qui n'est plus à nous : rien à lire.
                    raise DownloadRefused("file_gone", False)
                if response.status_code >= 400:
                    raise DownloadRefused(f"file_http_{response.status_code}", True)
                declared = int(response.headers.get("Content-Length") or 0)
                if declared > max_bytes:
                    raise DownloadRefused("file_too_large", False)
                with destination.open("wb") as handle:
                    for chunk in response.iter_bytes(65536):
                        written += len(chunk)
                        if written > max_bytes:
                            raise DownloadRefused("file_too_large", False)
                        handle.write(chunk)
        except httpx.HTTPError as exc:
            raise DownloadRefused("file_network", True) from exc
        return written
