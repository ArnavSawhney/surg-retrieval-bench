"""Open a dataset zip from a local path *or* its official URL, without unzipping it.

The Cholec80 archive is 74,916,858,781 bytes (69.8 GiB): larger than the free disk on
the development Mac. The CAMMA server supports HTTP range requests, so
``open_zip(url)`` reads the central directory and then only the members asked for.
``zipfile`` checks each member's CRC-32 when it is read to the end, so a member
streamed this way is verified exactly as one read from a local copy.

Where the archive lives is machine-specific and never tracked. Resolution order:
    1. env var ``SRB_<NAME>_ZIP`` (e.g. ``SRB_CHOLEC80_ZIP``),
    2. ``configs/local.yaml`` key ``<name>_zip`` (gitignored),
    3. otherwise raise, naming both options.
"""

from __future__ import annotations

import http.client
import io
import os
import shutil
import struct
import subprocess
import tempfile
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_CONFIG = REPO_ROOT / "configs" / "local.yaml"

OFFICIAL_URLS = {
    "cholec80": "https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.zip",
    "endoscapes": "https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip",
}
# Server Content-Length, read 2026-09-25 (Session 1) and re-read 2026-09-26.
EXPECTED_BYTES = {"cholec80": 74_916_858_781, "endoscapes": 6_285_499_749}


def resolve_zip(name: str) -> str:
    """Path or URL of dataset ``name``'s zip, from the env var or configs/local.yaml."""
    env = f"SRB_{name.upper()}_ZIP"
    if os.environ.get(env):
        return os.environ[env]
    if LOCAL_CONFIG.exists():
        cfg = yaml.safe_load(LOCAL_CONFIG.read_text()) or {}
        if cfg.get(f"{name}_zip"):
            return str(cfg[f"{name}_zip"])
    raise FileNotFoundError(
        f"Set {env}, or add '{name}_zip: <path or URL>' to configs/local.yaml "
        f"(gitignored). Official URL: {OFFICIAL_URLS.get(name, '?')}"
    )


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def source_size(source: str) -> int:
    """Byte size of a local file, or the Content-Length of a URL."""
    if is_url(source):
        req = urllib.request.Request(source, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers["Content-Length"])
    return Path(os.path.expanduser(source)).stat().st_size


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests (enough for ``zipfile``)."""

    def __init__(self, url: str, timeout: float = 120):
        self.url, self.timeout, self.pos = url, timeout, 0
        self.size = source_size(url)
        self.requests = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    MAX_REQUEST = 1 << 20  # short requests survive a slow server; callers loop

    def readinto(self, buf) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size, self.pos + self.MAX_REQUEST) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in range(6):  # the CAMMA server drops connections now and then
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    if r.status != 206:
                        raise OSError(f"server ignored the Range header (HTTP {r.status})")
                    data = r.read()
                break
            except http.client.IncompleteRead as e:
                if e.partial:  # keep what arrived; the caller asks again for the rest
                    data = e.partial
                    break
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        n = len(data)
        buf[:n] = data
        self.pos += n
        self.requests += 1
        return n


def open_zip(source: str) -> zipfile.ZipFile:
    """``ZipFile`` over a local path or an http(s) URL. Nothing is extracted."""
    if is_url(source):
        # 8 MiB buffer: few round trips when streaming a member.
        # Small buffer: listing and label files need only a few KB per read.
        return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(source), buffer_size=256 << 10))
    return zipfile.ZipFile(os.path.expanduser(source))


def _data_offset(source: str, info: zipfile.ZipInfo) -> int:
    """Byte offset of a member's compressed data (after its *local* file header,
    whose extra field can differ in length from the central directory's)."""
    if is_url(source):
        f = HTTPRangeFile(source)
        f.seek(info.header_offset)
        head = bytearray(30)
        f.readinto(head)
    else:
        with open(os.path.expanduser(source), "rb") as fh:
            fh.seek(info.header_offset)
            head = fh.read(30)
    sig, *_rest = struct.unpack("<4s26s", bytes(head))
    if sig != b"PK\x03\x04":
        raise OSError(f"bad local header signature for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", bytes(head[26:30]))
    return info.header_offset + 30 + name_len + extra_len


def _curl_range(url: str, start: int, end: int, out: Path) -> None:
    """Download bytes [start, end] to ``out``, resuming a partial file.

    Each curl call appends whatever arrives, contiguously, from ``start + have``; if it
    dies mid-way the next call resumes from the new file size. curl's own ``--retry``
    is deliberately NOT used: on a retry it restarts the same range and, writing to an
    append-mode stdout, would duplicate bytes already written.
    """
    want = end - start + 1
    have = out.stat().st_size if out.exists() else 0
    stalls = 0
    while have < want:
        cmd = ["curl", "-sS", "--fail", "--http1.1", "--connect-timeout", "30",
               "--speed-limit", "1000", "--speed-time", "60",
               "-r", f"{start + have}-{end}", url]
        with open(out, "ab") as fh:
            subprocess.run(cmd, stdout=fh, check=False)
        new = out.stat().st_size
        stalls = 0 if new > have else stalls + 1
        if stalls >= 50:
            raise OSError(f"could not fetch bytes {start}-{end} of {url} after retries")
        if new == have:
            time.sleep(min(60, 2 ** min(stalls, 6)))
        have = new
    if have != want:
        raise OSError(f"{out}: {have} bytes for a {want}-byte range (corrupt part)")


def fetch_member(source: str, zf: zipfile.ZipFile, name: str, dest: Path,
                 *, connections: int = 8, workdir: Path | None = None) -> dict:
    """Write member ``name`` to ``dest``, verifying its size and CRC-32.

    For a URL the member's compressed bytes are fetched as ``connections`` parallel
    range segments with ``curl`` (each resumable, so an interrupted run continues
    where it stopped), concatenated, then inflated while the CRC-32 is computed and
    compared with the zip's central directory. The compressed temp file is deleted
    afterwards. A local zip is read through ``zipfile`` (which checks the CRC itself).

    Returns ``{"compressed_bytes", "bytes", "crc32", "seconds"}``.
    """
    info = zf.getinfo(name)
    dest = Path(dest)
    t0 = time.time()
    if not is_url(source):
        with zf.open(info) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out, 16 << 20)
        return {"compressed_bytes": info.compress_size, "bytes": info.file_size,
                "crc32": info.CRC, "seconds": time.time() - t0}

    start = _data_offset(source, info)
    size = info.compress_size
    work = Path(workdir or tempfile.gettempdir()) / (Path(name).name + ".parts")
    work.mkdir(parents=True, exist_ok=True)
    n = max(1, min(connections, size // (1 << 20) or 1))
    bounds = [(start + size * i // n, start + size * (i + 1) // n - 1) for i in range(n)]
    parts = [work / f"part{i:03d}" for i in range(n)]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(n) as pool:
        list(pool.map(lambda a: _curl_range(source, a[0][0], a[0][1], a[1]),
                      zip(bounds, parts)))

    crc, total = 0, 0
    if info.compress_type == zipfile.ZIP_DEFLATED:
        inflater = zlib.decompressobj(-15)
    elif info.compress_type == zipfile.ZIP_STORED:
        inflater = None
    else:
        raise OSError(f"unsupported compression {info.compress_type} for {name}")
    partial = dest.with_name(dest.name + ".partial")  # renamed only once verified
    with open(partial, "wb") as out:
        for part in parts:
            with open(part, "rb") as fh:
                while chunk := fh.read(16 << 20):
                    data = inflater.decompress(chunk) if inflater else chunk
                    crc = zlib.crc32(data, crc)
                    total += len(data)
                    out.write(data)
        if inflater:
            tail = inflater.flush()
            crc, total = zlib.crc32(tail, crc), total + len(tail)
            out.write(tail)
    if total != info.file_size or crc != info.CRC:
        partial.unlink()
        shutil.rmtree(work)  # a corrupt segment must be re-fetched, not resumed
        raise OSError(f"{name}: size/CRC mismatch (got {total} bytes, crc {crc:08x}; "
                      f"expected {info.file_size}, {info.CRC:08x})")
    partial.rename(dest)
    shutil.rmtree(work)
    return {"compressed_bytes": size, "bytes": total, "crc32": crc,
            "seconds": time.time() - t0}
