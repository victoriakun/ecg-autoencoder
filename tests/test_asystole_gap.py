"""Smoke tests for the asystole gap detector."""
from realtime.asystole_gap import AsystoleGapDetector, scan


def test_no_gap_when_peaks_regular():
    fs = 360
    peaks = list(range(0, fs * 10, fs))  # 1 Hz, every second
    assert scan(peaks, fs=fs) == []


def test_severe_bradycardia_at_2s():
    fs = 360
    # peaks at 0s and 2.5s -> 2.5 s gap (>= severe brady 2.0 s, < asystole 3.0 s)
    events = scan([0, int(2.5 * fs)], fs=fs)
    assert len(events) == 1
    assert events[0].severity == "severe_bradycardia"
    assert 2.4 < events[0].gap_seconds < 2.6


def test_asystole_at_3s():
    fs = 360
    events = scan([0, int(3.5 * fs)], fs=fs)
    assert len(events) == 1
    assert events[0].severity == "asystole"
    assert 3.4 < events[0].gap_seconds < 3.6


def test_streaming_flush_when_stream_goes_silent():
    fs = 360
    det = AsystoleGapDetector(fs=fs)
    # one peak at t=0, then silence
    assert det.observe(0) == []
    # check at t=1.5s -> nothing
    assert det.flush(int(1.5 * fs)) == []
    # check at t=4s -> asystole open from 0 to 4s
    events = det.flush(int(4.0 * fs))
    assert len(events) == 1
    assert events[0].severity == "asystole"


def test_only_one_event_per_gap():
    fs = 360
    det = AsystoleGapDetector(fs=fs)
    det.observe(0)
    # One asystolic gap; the next observation closes it.
    events = det.observe(int(4 * fs))
    assert len(events) == 1
    assert events[0].severity == "asystole"
    # A subsequent normal beat must not emit anything
    assert det.observe(int(5 * fs)) == []
