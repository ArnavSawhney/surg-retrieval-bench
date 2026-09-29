"""zipsource: member fetch by byte range, inflate and CRC check.

The network is replaced by reads from a local zip (patching the two functions that
touch HTTP), so the URL code path -- local-header parsing, segmenting, concatenation,
inflate, CRC-32 -- is exercised without a server.
"""

from __future__ import annotations

import zipfile

import pytest

import srb.datasets.zipsource as zs


@pytest.fixture
def local_zip(tmp_path):
    path = tmp_path / "a.zip"
    payload = bytes(range(256)) * 20_000  # ~5 MB, compressible
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("videos/v.mp4", payload, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("stored.txt", b"hello" * 1000, compress_type=zipfile.ZIP_STORED)
    return path, payload


@pytest.fixture
def fake_url(monkeypatch, local_zip):
    path, _ = local_zip
    url = "https://example.invalid/a.zip"

    class FakeRange:
        def __init__(self, u):
            self.fh = open(path, "rb")

        def seek(self, o):
            self.fh.seek(o)

        def readinto(self, b):
            d = self.fh.read(len(b))
            b[: len(d)] = d
            return len(d)

    def fake_curl(u, start, end, out):
        with open(path, "rb") as fh:
            fh.seek(start)
            out.write_bytes(fh.read(end - start + 1))

    monkeypatch.setattr(zs, "HTTPRangeFile", FakeRange)
    monkeypatch.setattr(zs, "_curl_range", fake_curl)
    return url


@pytest.mark.parametrize("member,connections", [("videos/v.mp4", 1), ("videos/v.mp4", 4),
                                                ("stored.txt", 3)])
def test_fetch_member_url_path(tmp_path, local_zip, fake_url, member, connections):
    path, _ = local_zip
    zf = zipfile.ZipFile(path)
    dest = tmp_path / "out.bin"
    stats = zs.fetch_member(fake_url, zf, member, dest, connections=connections,
                            workdir=tmp_path / "work")
    assert dest.read_bytes() == zf.read(member)
    assert stats["crc32"] == zf.getinfo(member).CRC
    assert not (tmp_path / "work" / (dest.name + ".parts")).exists()
    assert not dest.with_name(dest.name + ".partial").exists()


def test_corrupt_bytes_fail_the_crc(tmp_path, local_zip, fake_url, monkeypatch):
    path, _ = local_zip
    zf = zipfile.ZipFile(path)
    info = zf.getinfo("stored.txt")

    def corrupt_curl(u, start, end, out):
        with open(path, "rb") as fh:
            fh.seek(start)
            data = bytearray(fh.read(end - start + 1))
        data[0] ^= 0xFF
        out.write_bytes(bytes(data))

    monkeypatch.setattr(zs, "_curl_range", corrupt_curl)
    dest = tmp_path / "out.bin"
    with pytest.raises(OSError, match="CRC"):
        zs.fetch_member(fake_url, zf, info.filename, dest, connections=1, workdir=tmp_path)
    assert not dest.exists()


def test_fetch_member_local_path(tmp_path, local_zip):
    path, payload = local_zip
    dest = tmp_path / "v.mp4"
    zs.fetch_member(str(path), zipfile.ZipFile(path), "videos/v.mp4", dest)
    assert dest.read_bytes() == payload


def test_resolve_zip_env_wins(monkeypatch):
    monkeypatch.setenv("SRB_CHOLEC80_ZIP", "/somewhere/cholec80.zip")
    assert zs.resolve_zip("cholec80") == "/somewhere/cholec80.zip"


def test_resolve_zip_missing_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("SRB_NOPE_ZIP", raising=False)
    monkeypatch.setattr(zs, "LOCAL_CONFIG", tmp_path / "missing.yaml")
    with pytest.raises(FileNotFoundError, match="SRB_NOPE_ZIP"):
        zs.resolve_zip("nope")


def test_curl_range_resumes_without_duplicating(tmp_path, monkeypatch):
    """A transfer that dies half-way is resumed from the exact byte count on disk.

    The fake curl serves the requested -r range but "drops" after 7 bytes on the first
    call; the result must still be the exact source bytes, with no duplication.
    """
    source = bytes(range(100))
    calls = []

    def fake_run(cmd, stdout, check):
        a, b = map(int, cmd[cmd.index("-r") + 1].split("-"))
        chunk = source[a:b + 1]
        if not calls:
            chunk = chunk[:7]
        calls.append((a, b))
        stdout.write(chunk)

    monkeypatch.setattr(zs.subprocess, "run", fake_run)
    out = tmp_path / "part"
    zs._curl_range("https://x.invalid/z", 10, 59, out)
    assert out.read_bytes() == source[10:60]
    assert calls == [(10, 59), (17, 59)]


def test_curl_range_detects_an_oversized_part(tmp_path):
    out = tmp_path / "part"
    out.write_bytes(b"x" * 20)
    with pytest.raises(OSError, match="corrupt part"):
        zs._curl_range("https://x.invalid/z", 0, 9, out)
