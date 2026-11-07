"""Tests for the local speech detection."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from custom_components.openai_whisper_cloud.speech import (
    SPEECH_FRAMES,
    VAD_FRAME_BYTES,
    SpeechDetector,
    wait_for_speech,
)

SILENT_FRAME = b"\x00" * VAD_FRAME_BYTES


async def chunks_of(*chunks: bytes) -> AsyncIterator[bytes]:
    """Yield the given chunks."""
    for chunk in chunks:
        yield chunk


def test_real_vad_hears_no_speech_in_silence() -> None:
    """microVAD itself finds no speech in three seconds of silence."""
    detector = SpeechDetector()
    assert not detector.process(SILENT_FRAME * 300)


def test_speech_needs_consecutive_frames(scripted_vad: Any) -> None:
    """A run of speech shorter than SPEECH_FRAMES does not count."""
    scripted_vad(
        [0.9] * (SPEECH_FRAMES - 1) + [0.1] + [0.9] * (SPEECH_FRAMES - 1) + [0.9]
    )
    detector = SpeechDetector()
    assert not detector.process(SILENT_FRAME * SPEECH_FRAMES)
    assert not detector.process(SILENT_FRAME * (SPEECH_FRAMES - 1))
    assert detector.process(SILENT_FRAME)


def test_unscored_frames_are_not_speech(scripted_vad: Any) -> None:
    """microVAD's -1 while it warms up is not speech."""
    scripted_vad([-1.0] * 50)
    assert not SpeechDetector().process(SILENT_FRAME * 50)


def test_frames_span_chunks(scripted_vad: Any) -> None:
    """Chunks that do not align with frames are scored frame by frame."""
    scripted_vad([0.9] * SPEECH_FRAMES)
    detector = SpeechDetector()
    audio = SILENT_FRAME * SPEECH_FRAMES
    pieces = [audio[offset : offset + 100] for offset in range(0, len(audio), 100)]
    for piece in pieces[:-1]:
        assert not detector.process(piece)
    assert detector.process(pieces[-1])


async def test_wait_for_speech_without_speech() -> None:
    """A stream that ends without speech gives None."""
    assert await wait_for_speech(chunks_of(SILENT_FRAME * 100, SILENT_FRAME)) is None


async def test_wait_for_speech_replays_from_the_start(scripted_vad: Any) -> None:
    """Once speech is heard, the whole stream comes back in order."""
    scripted_vad([0.0] * 100 + [0.9] * SPEECH_FRAMES)
    first = b"\x01" * VAD_FRAME_BYTES * 100
    second = b"\x02" * VAD_FRAME_BYTES * SPEECH_FRAMES
    third = b"\x03" * VAD_FRAME_BYTES
    stream = await wait_for_speech(chunks_of(first, second, third))
    assert stream is not None
    assert [chunk async for chunk in stream] == [first, second, third]
