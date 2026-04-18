"""Shared pytest fixtures for real-time pipeline tests."""
from __future__ import annotations

import sqlite3
import threading
from collections import deque
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest


class FastClock:
    """Deterministic clock for streaming tests. Advances only when tick() is called."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start
        self._lock = threading.Lock()

    def time(self) -> float:
        with self._lock:
            return self._t

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self._t += seconds

    def tick(self, seconds: float) -> None:
        self.sleep(seconds)


@pytest.fixture
def fast_clock() -> FastClock:
    return FastClock()


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "events.db"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
