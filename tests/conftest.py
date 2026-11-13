"""Aides des tests : réglages d'essai, faux Howé, images fabriquées.

Aucun test n'appelle un vrai Howé : le client est simulé, ou httpx passe par un
transport en mémoire. Tesseract, ZBar et OpenCV, eux, sont les vrais (légers,
sans modèle à charger).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw, ImageFont

from howe_worker.config import Settings

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        howe_url="http://howe.test",
        reference="00000000-0000-4000-8000-000000000001",
        seed=os.urandom(32),
        capabilities=("image",),
        slots={"image": 2},
        tmp_dir=tmp_path / "work",
        heartbeat_seconds=1,
    )


class FakeHowe:
    """Un Howé simulé : il sert les fichiers des tâches et note ce qu'on lui rend."""

    def __init__(self, files: dict[str, Path] | None = None, keep_going: bool = True):
        self.files = files or {}
        self.keep_going = keep_going
        self.completed: list[tuple[str, dict[str, Any]]] = []
        self.failed: list[tuple[str, str, bool]] = []
        self.metrics: list[tuple[str, dict[str, int] | None]] = []
        self.heartbeats = 0
        self.leases: list[list[dict[str, Any]]] = []
        self.stale = False
        self.uploaded: dict[str, int] = {}
        self.lease_lost = False

    def download(self, url: str, destination: Path, max_bytes: int) -> int:
        source = self.files[url]
        shutil.copyfile(source, destination)
        return source.stat().st_size

    def upload(self, reference: str, token: str, name: str, path: Path) -> None:
        if self.lease_lost:
            from howe_worker.client import LeaseLost

            raise LeaseLost(name)
        self.uploaded[name] = path.stat().st_size

    def heartbeat(self, reference: str, token: str, progress: int | None = None) -> bool:
        self.heartbeats += 1
        return self.keep_going

    def complete(self, reference: str, token: str, output: dict[str, Any], metrics: dict[str, int] | None = None) -> bool:
        self.completed.append((reference, output))
        self.metrics.append((reference, metrics))
        return not self.stale

    def fail(self, reference: str, token: str, code: str, retryable: bool, detail: str | None = None,
             metrics: dict[str, int] | None = None) -> None:
        self.failed.append((reference, code, retryable))
        self.metrics.append((reference, metrics))

    def lease(self, capabilities: list[str], slots: int, wait: int) -> list[dict[str, Any]]:
        return self.leases.pop(0) if self.leases else []


def task(reference: str = "t-1", url: str = "photo") -> dict[str, Any]:
    return {"reference": reference, "kind": "photo_check", "lease_token": "jeton",
            "input": {"read_url": url, "timeout": 60, "limits": {"max_side": 3840, "max_pixels": 40_000_000}}}


def sign_photo(path: Path, text: str, size: tuple[int, int] = (1200, 800), font_size: int = 64) -> Path:
    """Une photo fabriquée avec un texte écrit dessus, comme un panneau « À louer »."""
    image = Image.new("RGB", size, (236, 228, 214))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, font_size)
    draw.rectangle((60, 220, size[0] - 60, 220 + font_size * 3), fill="white", outline="black", width=4)
    draw.text((100, 240), text, fill="black", font=font)
    image.save(path)
    return path
