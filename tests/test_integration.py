"""Real models on synthetic audio. Opt-in: `pytest -m integration` (downloads weights once)."""

from pathlib import Path

import numpy as np
import pytest

from speech2song.backends.transcribe_whisper import WhisperTranscriber
from speech2song.config import AppConfig, DemucsConfig

from .fixtures.synth import noise_burst, speechlike, write_wav

pytestmark = pytest.mark.integration


def test_whisper_plumbing_on_synthetic_audio(tmp_path: Path) -> None:
    """Synthetic tones contain no words; this checks loading, decoding and conversion."""
    voice, _ = speechlike([(150, 250, 0.4), (250, 120, 0.4)] * 3)
    path = write_wav(tmp_path / "tones.wav", voice)
    config = AppConfig()
    result = WhisperTranscriber(config.whisper_model, "en", config.whisper).transcribe(path)
    assert result.language == "en"
    assert result.duration_s == pytest.approx(len(voice) / 44100, abs=0.05)
    assert all(w.start <= w.end for w in result.words())


@pytest.mark.parametrize("channels", [1, 2])
def test_demucs_keeps_shape(channels: int) -> None:
    from speech2song.backends.isolate_demucs import DemucsSeparator

    voice, _ = speechlike([(180, 220, 0.3), (220, 160, 0.3)] * 5)
    mixed = voice + noise_burst(len(voice) / 44100, amp=0.05)[: len(voice)]
    block = np.stack([mixed, mixed], axis=1) if channels == 2 else mixed[:, None]
    out = DemucsSeparator(DemucsConfig())(block.astype(np.float32))
    assert out.shape == block.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
