from pathlib import Path

from realtime.log_crypto import (
    ensure_key, encrypt_file, decrypt_file,
)


def test_key_is_generated_if_missing(tmp_path: Path):
    key_path = tmp_path / ".key"
    assert not key_path.exists()
    key = ensure_key(key_path)
    assert key_path.exists()
    assert len(key) == 44


def test_existing_key_is_returned(tmp_path: Path):
    key_path = tmp_path / ".key"
    k1 = ensure_key(key_path)
    k2 = ensure_key(key_path)
    assert k1 == k2


def test_roundtrip(tmp_path: Path):
    key_path = tmp_path / ".key"
    ensure_key(key_path)
    src = tmp_path / "sample.log"
    src.write_text("hello audit trail\n")
    enc = encrypt_file(src, key_path)
    assert enc.suffix == ".enc"
    assert not src.exists()

    out = tmp_path / "restored.log"
    decrypt_file(enc, key_path, out)
    assert out.read_text() == "hello audit trail\n"
