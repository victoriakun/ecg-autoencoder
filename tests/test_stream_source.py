import queue
import threading
import time

import numpy as np
import pytest

from realtime.stream_source import StreamSource, StreamWindow


@pytest.fixture
def fake_record(monkeypatch):
    """Patch wfdb.rdrecord so no actual file is read."""
    class FakeRec:
        p_signal = np.arange(5 * 360, dtype=float).reshape(-1, 1)
        fs = 360

    def _rdrecord(path):
        return FakeRec()

    import realtime.stream_source as ss
    monkeypatch.setattr(ss, "_rdrecord", _rdrecord)
    return FakeRec


def test_emits_windows_at_expected_count(fake_record, fast_clock):
    q: queue.Queue[StreamWindow] = queue.Queue()
    stop = threading.Event()
    src = StreamSource(
        patient_id="100",
        record_path="irrelevant",
        window_samples=720,
        stride_samples=180,
        clock=fast_clock,
        out_queue=q,
        stop_event=stop,
    )
    # Signal length = 5s * 360 = 1800 samples. Windows of 720 with stride 180 ->
    # (1800 - 720) / 180 + 1 = 7 windows.
    t = threading.Thread(target=src.run)
    t.start()

    # Advance fake clock enough for all windows (7 strides x 0.5 s)
    for _ in range(10):
        fast_clock.tick(0.5)
        time.sleep(0.001)  # let worker thread run

    stop.set()
    t.join(timeout=2.0)

    collected = []
    while not q.empty():
        collected.append(q.get_nowait())
    assert len(collected) == 7
    assert collected[0].patient_id == "100"
    assert collected[0].samples.shape == (720,)


def test_drops_short_windows(fake_record, fast_clock):
    """Signal length 700 (< 720) should emit zero windows with WARNING."""
    class ShortRec:
        p_signal = np.zeros((700, 1), dtype=float)
        fs = 360

    import realtime.stream_source as ss
    q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def _rdr(_):
        return ShortRec()
    ss._rdrecord = _rdr

    src = StreamSource("100", "x", 720, 180, fast_clock, q, stop)
    src.run()  # returns immediately
    assert q.empty()
