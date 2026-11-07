"""Local speech detection, so audio without speech is never transcribed.

Whisper models do not return an empty text for silence: they make one up,
usually a subtitle credit or a sentence of the prompt, which the prompt
conditions them to repeat. Home Assistant sends the STT provider everything
captured after the wake word until its VAD times out, so a wake word with
nothing after it reaches the provider as seconds of silence. The provider's
own ``no_speech_prob`` cannot be relied on to catch it: GroqCloud returns
0 for digital silence.

The audio is scored with microVAD, the same model the assist pipeline runs,
and held back until speech is heard.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import Final

from pymicro_vad import MicroVad

# microVAD scores 10 ms frames of 16 kHz, 16 bit mono PCM.
VAD_SAMPLE_RATE: Final = 16000
VAD_FRAME_BYTES: Final = 320

# A frame is speech above this probability. microVAD returns -1 until it has
# enough audio to score.
SPEECH_PROBABILITY: Final = 0.5
# Consecutive speech frames before the audio counts as speech. The assist
# pipeline's own segmenter wants 300 ms, which drops a short command like
# the Portuguese "para" (about 200 ms above 0.5); silence and steady noise
# never score a single frame.
SPEECH_FRAMES: Final = 10


class SpeechDetector:
    """Find the first run of speech in a stream of PCM chunks."""

    def __init__(self) -> None:
        """Initialize the detector."""
        self._vad = MicroVad()
        self._pending = b""
        self._speech_frames = 0

    def process(self, chunk: bytes) -> bool:
        """Score a chunk, return True once enough consecutive speech is heard."""
        self._pending += chunk
        offset = 0
        while offset + VAD_FRAME_BYTES <= len(self._pending):
            frame = self._pending[offset : offset + VAD_FRAME_BYTES]
            offset += VAD_FRAME_BYTES
            if self._vad.Process10ms(frame) > SPEECH_PROBABILITY:
                self._speech_frames += 1
                if self._speech_frames >= SPEECH_FRAMES:
                    return True
            else:
                self._speech_frames = 0
        self._pending = self._pending[offset:]
        return False


async def wait_for_speech(
    stream: AsyncIterable[bytes],
) -> AsyncIterator[bytes] | None:
    """Hold the audio back until speech is heard.

    Returns the whole stream, from its first chunk, once speech is heard, or
    None when the stream ends without any.
    """
    detector = SpeechDetector()
    held: list[bytes] = []
    chunks = aiter(stream)
    async for chunk in chunks:
        held.append(chunk)
        if detector.process(chunk):
            break
    else:
        return None

    async def replay() -> AsyncIterator[bytes]:
        for chunk in held:
            yield chunk
        async for chunk in chunks:
            yield chunk

    return replay()
