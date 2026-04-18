import queue
import threading

import numpy as np
import pytest
import torch

from realtime.inference import InferenceWorker, InferenceResult
from realtime.stream_source import StreamWindow


class IdentityModel(torch.nn.Module):
    """Returns input unchanged -> residual is always zero."""
    def forward(self, x):
        return x


class BiasedModel(torch.nn.Module):
    """Returns input + 1 -> residual = 1 for constant input of 0."""
    def forward(self, x):
        return x + 1.0


def test_identity_model_yields_zero_residual():
    in_q: queue.Queue[StreamWindow] = queue.Queue()
    out_q: queue.Queue[InferenceResult] = queue.Queue()
    stop = threading.Event()

    w = InferenceWorker(IdentityModel(), in_q, out_q, stop)
    in_q.put(StreamWindow("100", "t", np.zeros(720, dtype=float)))
    w.process_one(block=True, timeout=1.0)
    res = out_q.get_nowait()
    assert res.patient_id == "100"
    assert res.residual == pytest.approx(0.0, abs=1e-6)
    assert res.recon.shape == (720,)


def test_biased_model_yields_positive_residual():
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    stop = threading.Event()

    w = InferenceWorker(BiasedModel(), in_q, out_q, stop)
    in_q.put(StreamWindow("100", "t", np.zeros(720, dtype=float)))
    w.process_one(block=True, timeout=1.0)
    res = out_q.get_nowait()
    assert res.residual == pytest.approx(1.0, abs=1e-6)


def test_exception_in_forward_is_swallowed_and_logged(caplog):
    class Broken(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("boom")

    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    w = InferenceWorker(Broken(), in_q, out_q, stop)
    in_q.put(StreamWindow("100", "t", np.zeros(720, dtype=float)))
    with caplog.at_level("ERROR"):
        w.process_one(block=True, timeout=1.0)
    assert out_q.empty()
    assert "inference failed" in caplog.text.lower()
