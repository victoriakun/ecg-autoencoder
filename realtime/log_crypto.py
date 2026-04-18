"""Fernet encryption for rotated log files."""
from __future__ import annotations

import os
from pathlib import Path

from cryptography.fernet import Fernet


def ensure_key(path: Path) -> bytes:
    path = Path(path)
    if path.exists():
        return path.read_bytes()
    key = Fernet.generate_key()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return key


def encrypt_file(src: Path, key_path: Path) -> Path:
    key = Path(key_path).read_bytes()
    f = Fernet(key)
    src = Path(src)
    data = src.read_bytes()
    enc_path = src.with_suffix(src.suffix + ".enc")
    enc_path.write_bytes(f.encrypt(data))
    try:
        enc_path.chmod(0o600)
    except OSError:
        pass
    src.unlink()
    return enc_path


def decrypt_file(enc_path: Path, key_path: Path, out_path: Path) -> None:
    key = Path(key_path).read_bytes()
    f = Fernet(key)
    enc_path = Path(enc_path)
    data = f.decrypt(enc_path.read_bytes())
    out_path = Path(out_path)
    out_path.write_bytes(data)
