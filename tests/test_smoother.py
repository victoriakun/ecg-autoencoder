from realtime.smoother import NofMSmoother


def test_initial_state_is_false():
    s = NofMSmoother(k=2, m=3)
    assert s.confirmed is False


def test_single_exceedance_not_confirmed():
    s = NofMSmoother(k=2, m=3)
    s.push(True)
    assert s.confirmed is False


def test_two_of_three_confirms():
    s = NofMSmoother(k=2, m=3)
    s.push(True)
    s.push(False)
    s.push(True)
    assert s.confirmed is True


def test_falling_edge_after_enough_negatives():
    s = NofMSmoother(k=2, m=3)
    for v in [True, True, True]:
        s.push(v)
    assert s.confirmed is True
    s.push(False)
    s.push(False)
    # Now window is [True, False, False] -> confirmed = False
    assert s.confirmed is False


def test_rising_edge_event_reported_once():
    s = NofMSmoother(k=2, m=3)
    assert s.push(False) == "none"
    assert s.push(True) == "none"
    assert s.push(True) == "rising"       # first time confirmed
    assert s.push(True) == "none"         # still confirmed, no new edge
    assert s.push(False) == "none"
    assert s.push(False) == "falling"     # de-confirmed


def test_k_equals_m():
    s = NofMSmoother(k=3, m=3)
    s.push(True); s.push(True); s.push(False)
    assert s.confirmed is False
    s.push(True); s.push(True); s.push(True)
    assert s.confirmed is True
