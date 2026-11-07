"""Tests for the local speech detection."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from homeassistant.components.assist_pipeline.vad import VoiceCommandSegmenter

from custom_components.openai_whisper_cloud.speech import (
    VAD_FRAME_BYTES,
    VAD_FRAME_SECONDS,
    SpeechDetector,
    wait_for_speech,
)

SILENT_FRAME = b"\x00" * VAD_FRAME_BYTES
# Speech as the pipeline's segmenter counts it before a command starts
SPEECH = VoiceCommandSegmenter.before_command_speech_threshold + 0.05
# Frames of speech that start a command, one more against rounding
COMMAND_FRAMES = round(VoiceCommandSegmenter.speech_seconds / VAD_FRAME_SECONDS) + 1
TIMEOUT_FRAMES = round(VoiceCommandSegmenter.timeout_seconds / VAD_FRAME_SECONDS)


async def chunks_of(*chunks: bytes) -> AsyncIterator[bytes]:
    """Yield the given chunks."""
    for chunk in chunks:
        yield chunk


def test_real_vad_hears_no_command_in_silence() -> None:
    """microVAD itself finds no speech in three seconds of silence."""
    detector = SpeechDetector()
    assert not detector.process(SILENT_FRAME * 300)


def test_command_starts_as_the_pipeline_starts_it(scripted_vad: Any) -> None:
    """A command starts after the pipeline's speech_seconds of speech."""
    scripted_vad([SPEECH] * COMMAND_FRAMES)
    detector = SpeechDetector()
    assert not detector.process(SILENT_FRAME * (COMMAND_FRAMES - 5))
    assert detector.process(SILENT_FRAME * 5)


def test_scores_under_the_threshold_are_not_speech(scripted_vad: Any) -> None:
    """Frames at the pipeline's threshold never start a command."""
    threshold = VoiceCommandSegmenter.before_command_speech_threshold
    scripted_vad([threshold] * 100)
    assert not SpeechDetector().process(SILENT_FRAME * 100)


def test_unscored_frames_are_not_speech(scripted_vad: Any) -> None:
    """microVAD's -1 while it warms up is not speech."""
    scripted_vad([-1.0] * 50)
    assert not SpeechDetector().process(SILENT_FRAME * 50)


def test_frames_span_chunks(scripted_vad: Any) -> None:
    """Chunks that do not align with frames are scored frame by frame."""
    scripted_vad([SPEECH] * COMMAND_FRAMES)
    whole_frames = SpeechDetector()
    frames = 1
    while not whole_frames.process(SILENT_FRAME):
        frames += 1

    detector = SpeechDetector()
    audio = SILENT_FRAME * COMMAND_FRAMES
    pieces = [audio[offset : offset + 100] for offset in range(0, len(audio), 100)]
    first = next(i for i, piece in enumerate(pieces) if detector.process(piece))
    # the command starts with the piece that completes its last frame
    assert first * 100 < frames * VAD_FRAME_BYTES <= (first + 1) * 100


async def test_wait_for_speech_timed_out_without_a_command() -> None:
    """A stream the pipeline timed out on without a command gives None."""
    # the pipeline keeps back the frame it times out on
    frames = [SILENT_FRAME] * (TIMEOUT_FRAMES - 1)
    assert await wait_for_speech(chunks_of(*frames)) is None


async def test_wait_for_speech_ended_early_without_a_command() -> None:
    """A stream that ends before the timeout is uploaded, a command or not."""
    frames = [SILENT_FRAME] * (TIMEOUT_FRAMES - 2)
    stream = await wait_for_speech(chunks_of(*frames))
    assert stream is not None
    assert [chunk async for chunk in stream] == frames


async def test_wait_for_speech_empty_stream() -> None:
    """An empty stream gives None."""
    assert await wait_for_speech(chunks_of()) is None


async def test_wait_for_speech_replays_from_the_start(scripted_vad: Any) -> None:
    """Once a command starts, the whole stream comes back in order."""
    scripted_vad([0.0] * 100 + [SPEECH] * COMMAND_FRAMES)
    first = b"\x01" * VAD_FRAME_BYTES * 100
    second = b"\x02" * VAD_FRAME_BYTES * COMMAND_FRAMES
    third = b"\x03" * VAD_FRAME_BYTES
    stream = await wait_for_speech(chunks_of(first, second, third))
    assert stream is not None
    assert [chunk async for chunk in stream] == [first, second, third]
