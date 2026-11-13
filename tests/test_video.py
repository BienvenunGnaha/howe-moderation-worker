"""La vidéo avec les vrais FFmpeg, Tesseract et (une fois) Faster-Whisper, sur des
vidéos fabriquées : un numéro qui s'affiche de 2 à 5 s, une vidéo trop longue, un
fichier sans image, des fichiers piégés. Howé est simulé."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from howe_worker import video_frames_read, video_prepare, video_transcribe
from howe_worker.config import Settings
from howe_worker.photo_check import Cancelled, TaskFailed
from howe_worker.runner import Runner

from .conftest import FONT, FakeHowe

INPUT = {"read_url": "video", "timeout": 240, "limits": {"max_video_seconds": 45.5, "max_side": 3840, "max_pixels": 40_000_000},
         "publish": {"max_side": 1280, "max_fps": 30, "crf": 23, "max_bitrate": "2500k", "audio_bitrate": "96k", "poster_at": 1.0},
         "frames": {"per_second": 2, "scene": 0.3, "max": 120}}


def make_clip(path: Path, seconds: float = 6, text: str | None = None, audio: bool = True, size: str = "640x360", rate: int = 15) -> Path:
    """Une vidéo fabriquée : fond uni, texte affiché de 2 à 5 s, son de 440 Hz (sans paroles)."""
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x2f6b4f:s={size}:r={rate}:d={seconds}"]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    if text:
        command += ["-vf", f"drawtext=fontfile={FONT}:text='{text}':fontsize=44:fontcolor=white:box=1:boxcolor=black:x=40:y=150"
                           ":enable='between(t,2,5)'"]
    command += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-metadata", "location=+06.36+002.39/"]
    command += (["-c:a", "aac", "-shortest"] if audio else []) + [str(path)]
    subprocess.run(command, check=True)
    return path


def prepare(settings: Settings, source: Path, keep_going=lambda: True, howe: FakeHowe | None = None) -> tuple[dict, Path, FakeHowe]:
    workdir = settings.tmp_dir / "prepare"
    workdir.mkdir(parents=True, exist_ok=True)
    howe = howe or FakeHowe({"video": source})
    task = {"reference": "t-video", "lease_token": "jeton", "kind": "video_prepare", "input": INPUT}
    output = video_prepare.run(INPUT, workdir, howe, settings, keep_going, task=task)  # type: ignore[arg-type]
    return output, workdir, howe


def probe(path: Path) -> dict:
    return json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                                     capture_output=True, text=True, check=True).stdout)


def test_a_video_is_prepared_for_publication(settings: Settings, tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "salon.mp4", text="A LOUER 97 45 32 21")

    output, workdir, howe = prepare(settings, clip)

    assert 5800 <= output["duration_ms"] <= 6200 and output["has_audio"] is True
    assert {"published.mp4", "poster.webp", "audio.wav"} <= set(output["files"])
    assert set(howe.uploaded) == set(output["files"])
    published = probe(workdir / "published.mp4")
    video = next(s for s in published["streams"] if s["codec_type"] == "video")
    audio = next(s for s in published["streams"] if s["codec_type"] == "audio")
    assert (video["codec_name"], video["profile"], audio["codec_name"]) == ("h264", "High", "aac")
    # Métadonnées retirées : la position du téléphone ne part jamais avec la vidéo publiée.
    assert "location" not in json.dumps(published["format"].get("tags", {}))
    wav = probe(workdir / "audio.wav")["streams"][0]
    assert (wav["sample_rate"], wav["channels"]) == ("16000", 1)
    # Le texte apparaît à 2 s et disparaît à 5 s : ces deux moments sont gardés, les images identiques écartées.
    times = [frame["at_ms"] for frame in output["frames"]]
    assert any(1900 <= t <= 2600 for t in times) and len(times) < 12


def test_text_shown_in_the_video_is_read_with_its_time(settings: Settings, tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "panneau.mp4", text="A LOUER 97 45 32 21")
    prepared, workdir, _ = prepare(settings, clip)
    files = {f"url-{frame['name']}": workdir / frame["name"] for frame in prepared["frames"]}
    frames = [{"name": f["name"], "at_ms": f["at_ms"], "url": f"url-{f['name']}"} for f in prepared["frames"]]

    read_dir = settings.tmp_dir / "read"
    read_dir.mkdir()
    output = video_frames_read.run({"frames": frames, "timeout": 300, "limits": INPUT["limits"]}, read_dir, FakeHowe(files),
                                   settings, lambda: True)  # type: ignore[arg-type]

    seen = [frame["at_ms"] for frame in output["frames"] if any("97453221" in re.sub(r"\D", "", line["text"]) for line in frame["text"])]
    assert seen and 1900 <= min(seen) <= 2600
    assert output["metrics"]["frames_read"] == len(frames)


def test_a_video_longer_than_45_seconds_is_refused_before_encoding(settings: Settings, tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "longue.mp4", seconds=46, audio=False, size="160x120", rate=5)

    with pytest.raises(TaskFailed) as caught:
        prepare(settings, clip)
    assert (caught.value.code, caught.value.retryable) == ("video_too_long", False)


def test_a_file_without_images_is_refused(settings: Settings, tmp_path: Path) -> None:
    sound = tmp_path / "son.m4a"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=duration=3", "-c:a", "aac", str(sound)], check=True)

    with pytest.raises(TaskFailed) as caught:
        prepare(settings, sound)
    assert (caught.value.code, caught.value.retryable) == ("video_no_stream", False)


def test_broken_and_disguised_files_are_refused(settings: Settings, tmp_path: Path) -> None:
    broken = tmp_path / "abimee.mp4"
    broken.write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 40)
    with pytest.raises(TaskFailed) as caught:
        prepare(settings, broken)
    assert (caught.value.code, caught.value.retryable) == ("video_unreadable", False)

    # Une liste de lecture HLS déguisée en MP4 (elle lirait /etc/passwd) : refusée sans être ouverte.
    playlist = tmp_path / "piege.mp4"
    playlist.write_text("#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:10.0,\nfile:///etc/passwd\n#EXT-X-ENDLIST\n")
    with pytest.raises(TaskFailed) as caught:
        prepare(settings, playlist)
    assert caught.value.code == "file_unsupported"


def test_a_cancelled_preparation_stops(settings: Settings, tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "salon.mp4")
    with pytest.raises(Cancelled):
        prepare(settings, clip, keep_going=lambda: False)


def test_a_lost_lease_while_uploading_stops_without_result(settings: Settings, tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "salon.mp4", seconds=3)
    howe = FakeHowe({"video": clip})
    howe.lease_lost = True
    task = {"reference": "t-video", "lease_token": "jeton", "kind": "video_prepare", "input": INPUT}

    assert Runner(settings, howe).process(task) == "cancelled"  # type: ignore[arg-type]
    assert howe.completed == [] and list(settings.tmp_dir.iterdir()) == []


def test_speech_segments_are_returned_and_cancellation_is_honoured(settings: Settings, tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    class Segment:
        def __init__(self, start: float, end: float, text: str):
            self.start, self.end, self.text = start, end, text

    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            info = type("Info", (), {"language": "fr"})()
            return iter([Segment(9.0, 12.0, " Appelez le zéro un soixante-sept"), Segment(12.0, 16.0, " quarante-cinq trente-deux vingt et un.")]), info

    monkeypatch.setattr(video_transcribe, "model", lambda _settings: FakeModel())
    import wave

    audio = tmp_path / "son.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)
    howe = FakeHowe({"audio": audio})

    output = video_transcribe.run({"audio_url": "audio", "timeout": 360}, tmp_path, howe, settings, lambda: True)  # type: ignore[arg-type]
    assert output["language"] == "fr"
    assert output["segments"][0] == {"start_ms": 9000, "end_ms": 12000, "text": "Appelez le zéro un soixante-sept"}

    with pytest.raises(Cancelled):
        video_transcribe.run({"audio_url": "audio", "timeout": 360}, tmp_path, howe, settings, lambda: False)  # type: ignore[arg-type]


def _whisper_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache("Systran/faster-whisper-small", "model.bin"), str)
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _whisper_cached(), reason="modèle Whisper small absent du cache (il n'est jamais téléchargé par les tests)")
def test_the_real_whisper_model_transcribes_a_clip(settings: Settings, tmp_path: Path) -> None:
    """Le vrai Faster-Whisper small (autorisé sur le poste le 13/11/2026), sur 6 s sans paroles."""
    clip = make_clip(tmp_path / "salon.mp4")
    _, workdir, _ = prepare(settings, clip)
    howe = FakeHowe({"audio": workdir / "audio.wav"})

    output = video_transcribe.run({"audio_url": "audio", "timeout": 360}, tmp_path, howe, settings, lambda: True)  # type: ignore[arg-type]

    assert output["model"] == "small" and output["engines"][0].startswith("faster-whisper")
    assert isinstance(output["language"], str)
    # Un son pur n'a pas de paroles : rien, ou presque, et surtout aucun chiffre inventé.
    assert not any(re.search(r"\d{4,}", segment["text"]) for segment in output["segments"])
