import os
import queue
import threading
import time

import numpy as np
import pytest
import torch

from realtime.inference import InferenceResult, InferenceWorker, load_model
from realtime.stream_source import StreamWindow
from models import ConvAutoencoder


pytestmark = pytest.mark.latency


@pytest.mark.skipif(
    os.environ.get("TORCH_CPU_BENCHMARK") != "1",
    reason="set TORCH_CPU_BENCHMARK=1 to run the latency benchmark",
)
def test_p95_latency_under_100ms():
    from pathlib import Path
    model = load_model(Path("models/ecg_autoencoder.pt"), ConvAutoencoder)

    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    worker = InferenceWorker(model, in_q, out_q, stop, sampling_rate=360)

    N = 100
    for _ in range(N):
        in_q.put(StreamWindow("100", "t", np.random.randn(720).astype(float)))

    latencies = []
    for _ in range(N):
        start = time.perf_counter()
        worker.process_one(block=True, timeout=5.0)
        latencies.append((time.perf_counter() - start) * 1000)

    p95 = float(np.quantile(latencies, 0.95))
    print(f"inference p95 = {p95:.1f} ms")
    assert p95 < 100.0, f"p95 {p95:.1f} ms exceeds 100 ms budget"
