"""photo_check avec les vrais moteurs (Tesseract, ZBar, OpenCV), sur des photos
fabriquées : un numéro écrit sur un panneau, un QR code, des fichiers hostiles."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import segno
from PIL import Image

from howe_worker import photo_check
from howe_worker.config import Settings
from howe_worker.photo_check import Cancelled, TaskFailed

from .conftest import FakeHowe, sign_photo, task


def run(settings: Settings, path: Path, keep_going=lambda: True) -> dict:
    workdir = settings.tmp_dir / "t"
    workdir.mkdir(parents=True, exist_ok=True)
    howe = FakeHowe({"photo": path})
    return photo_check.run(task()["input"], workdir, howe, settings, keep_going)  # type: ignore[arg-type]


def digits(output: dict) -> str:
    return " | ".join(re.sub(r"\D", "", line["text"]) for line in output["text"])


def test_a_phone_written_on_a_sign_is_read(settings: Settings, tmp_path: Path) -> None:
    photo = sign_photo(tmp_path / "panneau.png", "A LOUER 97 45 32 21")

    output = run(settings, photo)

    assert "97453221" in digits(output)
    line = next(l for l in output["text"] if "97453221" in re.sub(r"\D", "", l["text"]))
    assert line["confidence"] >= 60
    assert {l["version"] for l in output["text"]} == {"original", "contrast"}
    assert output["engines"][0].startswith("tesseract")
    assert output["metrics"]["width"] == 1200


def test_a_small_number_is_read_on_the_enlarged_version(settings: Settings, tmp_path: Path) -> None:
    # Écrit petit, sur une petite image : la version agrandie et contrastée le retrouve.
    photo = sign_photo(tmp_path / "petit.png", "Tel 01 67 45 32 21", size=(640, 420), font_size=22)

    output = run(settings, photo)

    assert "0167453221" in digits(output)


def test_a_qr_code_is_decoded(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "qr.png"
    segno.make("https://wa.me/22997453221", error="m").save(path, scale=8, border=4)

    output = run(settings, path)

    assert output["qr"] == [{"data": "https://wa.me/22997453221", "engine": "zbar"}]


def test_opencv_reads_what_zbar_missed(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "qr.png"
    segno.make("tel:+22997453221", error="m").save(path, scale=8, border=4)
    from pyzbar import pyzbar

    monkeypatch.setattr(pyzbar, "decode", lambda *a, **k: [])

    output = run(settings, path)

    assert output["qr"] == [{"data": "tel:+22997453221", "engine": "opencv"}]


def test_scattered_words_in_a_scene_are_read(settings: Settings, tmp_path: Path) -> None:
    from PIL import ImageDraw, ImageFont

    from .conftest import FONT

    scene = Image.new("RGB", (1000, 700), (90, 120, 80))
    draw = ImageDraw.Draw(scene)
    draw.text((600, 80), "WA 96 12 34 56", fill="white", font=ImageFont.truetype(FONT, 40))
    draw.text((60, 560), "Cotonou", fill="yellow", font=ImageFont.truetype(FONT, 40))
    scene.save(tmp_path / "scene.png")

    output = run(settings, tmp_path / "scene.png")

    assert "96123456" in digits(output)


def test_a_clean_photo_gives_no_number(settings: Settings, tmp_path: Path) -> None:
    photo = tmp_path / "salon.png"
    Image.new("RGB", (800, 600), (120, 160, 200)).save(photo)

    output = run(settings, photo)

    assert digits(output).replace("|", "").strip() == ""
    assert output["qr"] == []


def test_a_file_that_is_not_an_image_is_refused(settings: Settings, tmp_path: Path) -> None:
    fake = tmp_path / "photo.jpg"
    fake.write_bytes(b"#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\nfile:///etc/passwd\n")

    with pytest.raises(TaskFailed) as caught:
        run(settings, fake)
    assert (caught.value.code, caught.value.retryable) == ("file_unsupported", False)


def test_a_decompression_bomb_is_refused_before_decoding(settings: Settings, tmp_path: Path) -> None:
    bomb = tmp_path / "bombe.png"
    Image.new("1", (9000, 9000)).save(bomb)  # 81 mégapixels pour quelques kilo-octets

    with pytest.raises(TaskFailed) as caught:
        run(settings, bomb)
    assert caught.value.code == "image_too_large"


def test_a_truncated_image_is_refused(settings: Settings, tmp_path: Path) -> None:
    photo = sign_photo(tmp_path / "entiere.png", "A LOUER")
    cut = tmp_path / "tronquee.png"
    cut.write_bytes(photo.read_bytes()[:400])

    with pytest.raises(TaskFailed) as caught:
        run(settings, cut)
    assert (caught.value.code, caught.value.retryable) == ("file_unreadable", False)


def test_a_cancelled_task_stops(settings: Settings, tmp_path: Path) -> None:
    photo = sign_photo(tmp_path / "panneau.png", "A LOUER 97 45 32 21")

    with pytest.raises(Cancelled):
        run(settings, photo, keep_going=lambda: False)


def test_a_huge_phone_photo_is_reduced_not_refused(settings: Settings, tmp_path: Path) -> None:
    photo = tmp_path / "grande.jpg"
    Image.new("RGB", (4032, 3024), (200, 200, 200)).save(photo, quality=60)

    output = run(settings, photo)

    assert max(output["metrics"]["width"], output["metrics"]["height"]) == 3840
