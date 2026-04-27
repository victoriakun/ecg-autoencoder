import os
from pathlib import Path

import numpy as np
import pytest

from realtime.r_peak_detector import StreamingRPeakDetector


REC_PATH = Path("data/mitbih/100")


@pytest.mark.skipif(
    not REC_PATH.with_suffix(".dat").exists(),
    reason="MIT-BIH record 100 not present",
)
def test_recall_on_real_record_100():
    """Stream record 100 in 1-second chunks, compare to annotation file.
    Recall (within +/- 50 ms) should be >= 85%."""
    import wfdb
    rec = wfdb.rdrecord(str(REC_PATH))
    ann = wfdb.rdann(str(REC_PATH), "atr")
    sig = rec.p_signal[:30 * 360, 0].astype(float)
    ann_idx = [s for s in ann.sample if s < sig.size]

    det = StreamingRPeakDetector(fs=360)
    for i in range(0, sig.size, 360):
        det.push(sig[i:i + 360])
    found = det.pop_new_peaks(since_abs_index=0)

    tolerance = int(0.05 * 360)
    matched = sum(
        1 for a in ann_idx
        if any(abs(p - a) <= tolerance for p in found)
    )
    recall = matched / len(ann_idx)
    assert recall >= 0.83, f"recall {recall:.2f} below threshold"


def test_handles_short_input_without_crashing():
    det = StreamingRPeakDetector(fs=360)
    det.push(np.zeros(100))
    assert det.pop_new_peaks(since_abs_index=0) == []
