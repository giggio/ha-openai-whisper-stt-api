"""Local speech detection, so audio without speech is never transcribed.

Whisper models do not return an empty text for silence: they make one up,
usually a subtitle credit or a sentence of the prompt, which the prompt
conditions them to repeat. Home Assistant sends the STT provider everything
captured after the wake word until its VAD times out, so a wake word with
nothing after it reaches the provider as seconds of silence. The provider's
own ``no_speech_prob`` cannot be relied on to catch it: GroqCloud returns
0 for digital silence.

The audio is held back until the assist pipeline's own voice command
segmenter, fed by microVAD as the pipeline feeds it, hears a command start.
The pipeline scores the same audio it sends here (it applies no noise
suppression or gain for ESPHome satellites), so that is its own verdict.
A stream that ends before the pipeline's timeout is uploaded even without
one: only the pipeline hearing a command, or a satellite hearing its talker
stop, ends it that soon. Audio is dropped only when it runs into the
timeout without a command, which is what silence after a wake word does.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import Final

from homeassistant.components.assist_pipeline.vad import VoiceCommandSegmenter
from pymicro_vad import MicroVad

# microVAD scores 10 ms frames of 16 kHz, 16 bit mono PCM.
VAD_SAMPLE_RATE: Final = 16000
VAD_FRAME_BYTES: Final = 320
VAD_FRAME_SECONDS: Final = 0.01
VAD_BYTES_PER_SECOND: Final = VAD_SAMPLE_RATE * 2


class SpeechDetector:
    """Find where a voice command starts in a stream of PCM chunks."""

    def __init__(self) -> None:
        """Initialize the detector."""
        self._vad = MicroVad()
        self._segmenter = VoiceCommandSegmenter()
        self._pending = b""

    def process(self, chunk: bytes) -> bool:
        """Score a chunk, return True once a voice command has started."""
        self._pending += chunk
        offset = 0
        while offset + VAD_FRAME_BYTES <= len(self._pending):
            frame = self._pending[offset : offset + VAD_FRAME_BYTES]
            offset += VAD_FRAME_BYTES
            self._segmenter.process(VAD_FRAME_SECONDS, self._vad.Process10ms(frame))
            if self._segmenter.in_command:
                return True
        self._pending = self._pending[offset:]
        return False


async def wait_for_speech(
    stream: AsyncIterable[bytes],
) -> AsyncIterator[bytes] | None:
    """Hold the audio back until it is known to carry speech.

    Returns the whole stream, from its first chunk, once a voice command
    starts or the stream ends before the pipeline's timeout. Returns None when
    the stream is empty, or ran into the timeout without a command.
    """
    detector = SpeechDetector()
    held: list[bytes] = []
    size = 0
    chunks = aiter(stream)
    async for chunk in chunks:
        held.append(chunk)
        size += len(chunk)
        if detector.process(chunk):
            break
    else:
        # The pipeline keeps back the chunk it timed out on, as long as the
        # ones before it.
        timeout_bytes = VoiceCommandSegmenter.timeout_seconds * VAD_BYTES_PER_SECOND
        if not held or size + len(held[-1]) >= timeout_bytes:
            return None

    async def replay() -> AsyncIterator[bytes]:
        for chunk in held:
            yield chunk
        async for chunk in chunks:
            yield chunk

    return replay()
