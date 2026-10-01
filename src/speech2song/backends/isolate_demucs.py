"""demucs voice isolation (optional `isolate` extra: `uv sync --extra isolate`)."""

import logging

import numpy as np

from speech2song.config import SAMPLE_RATE, DemucsConfig
from speech2song.errors import S2SError

log = logging.getLogger(__name__)


class DemucsSeparator:
    """Callable separator returning the vocals stem with the input's channel count."""

    def __init__(self, config: DemucsConfig) -> None:
        try:
            import demucs.api
            import torch
        except ImportError as exc:
            raise S2SError(
                "Voice isolation needs the optional demucs dependencies: "
                "run `uv sync --extra isolate`."
            ) from exc
        self._torch = torch
        log.info("Loading demucs model %s on %s", config.model, config.device)
        self._separator = demucs.api.Separator(
            model=config.model,
            device=config.device,
            shifts=config.shifts,  # 0 = deterministic (shifts adds random time offsets)
            overlap=config.overlap,
            jobs=config.jobs,
            progress=False,
        )
        if self._separator.samplerate != SAMPLE_RATE:
            raise S2SError(
                f"demucs model {config.model} runs at {self._separator.samplerate} Hz; "
                f"the pipeline uses {SAMPLE_RATE} Hz."
            )
        if "vocals" not in self._separator.model.sources:
            raise S2SError(f"demucs model {config.model} has no 'vocals' stem.")
        self._channels = self._separator.audio_channels

    def __call__(self, block: np.ndarray) -> np.ndarray:
        frames, channels = block.shape
        if channels not in (1, 2):
            raise S2SError(f"Expected mono or stereo audio, got {channels} channels.")
        model_in = block if channels == self._channels else np.repeat(block[:, :1], 2, axis=1)
        wav = self._torch.from_numpy(np.ascontiguousarray(model_in.T, dtype=np.float32))
        with self._torch.inference_mode():
            _, stems = self._separator.separate_tensor(wav, SAMPLE_RATE)
        vocals = stems["vocals"].cpu().numpy().T.astype(np.float32)
        if channels == 1:
            vocals = vocals.mean(axis=1, keepdims=True)
        if vocals.shape != (frames, channels):
            raise S2SError(f"demucs returned shape {vocals.shape}, expected {(frames, channels)}")
        return vocals
