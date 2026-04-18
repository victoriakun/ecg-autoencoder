"""Inference worker: consumes stream windows, emits residuals."""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from preprocess import preprocess
from realtime.stream_source import StreamWindow

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    patient_id: str
    ts_utc: str
    raw: np.ndarray
    recon: np.ndarray
    residual: float


def load_model(path: Path, build_fn) -> nn.Module:
    """Load trained weights into the model returned by `build_fn()`.

    If the checkpoint stores `latent_dim` / `input_dim` at the top level,
    pass them as kwargs to `build_fn` so architectures that differ only in
    those hyperparameters are reconstructed correctly.
    """
    ckpt = torch.load(path, map_location="cpu")
    kwargs = {}
    if isinstance(ckpt, dict):
        for k in ("input_dim", "latent_dim"):
            if k in ckpt and isinstance(ckpt[k], int):
                kwargs[k] = ckpt[k]

    state = ckpt
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in state:
                state = state[key]
                break

    # Infer last-conv output width from the checkpoint's encoder.9 layer so
    # the same build_fn can load different ConvAutoencoder variants.
    enc9 = state.get("encoder.9.weight") if isinstance(state, dict) else None
    if enc9 is not None and enc9.dim() >= 1:
        kwargs["last_channels"] = int(enc9.shape[0])

    try:
        model = build_fn(**kwargs)
    except TypeError:
        kwargs.pop("last_channels", None)
        try:
            model = build_fn(**kwargs)
        except TypeError:
            model = build_fn()
    model.load_state_dict(state)
    model.eval()
    return model


class InferenceWorker:
    def __init__(
        self,
        model: nn.Module,
        in_queue: "queue.Queue[StreamWindow]",
        out_queue: "queue.Queue[InferenceResult]",
        stop_event: threading.Event,
        sampling_rate: int = 360,
    ) -> None:
        self._model = model
        self._in = in_queue
        self._out = out_queue
        self._stop = stop_event
        self._fs = sampling_rate

    def run(self) -> None:
        while not self._stop.is_set():
            self.process_one(block=True, timeout=0.5)

    def process_one(self, *, block: bool, timeout: float) -> None:
        try:
            win = self._in.get(block=block, timeout=timeout)
        except queue.Empty:
            return
        try:
            pre = preprocess(win.samples, fs=self._fs, apply_bandpass=True,
                             apply_notch=False)
            x = torch.from_numpy(pre).float().unsqueeze(0)
            with torch.no_grad():
                recon_t = self._model(x)
            recon = recon_t.squeeze(0).cpu().numpy().astype(float)
            residual = float(np.mean((pre - recon) ** 2))
            self._out.put(
                InferenceResult(
                    patient_id=win.patient_id,
                    ts_utc=win.ts_utc,
                    raw=pre,
                    recon=recon,
                    residual=residual,
                ),
                timeout=1.0,
            )
        except Exception as e:
            log.error("inference failed for %s: %s", win.patient_id, e)
