"""
Deepfake Detection System (Image)
==================================
Implements deepfake detection with robust OS-level components:
  - Multiprocessing via fork/exec/wait patterns
  - POSIX shared memory via mmap for inter-process image buffers
  - POSIX semaphores (threading.Semaphore) for worker synchronization
  - Buffered vs direct I/O comparison
  - fsync-safe result logging
  - Proper error handling on all system calls

Dataset format follows COCO-style JSON:
  { "categories": [{"id":0,"name":"Real"},{"id":1,"name":"Fake"}],
    "images": [{"id":0,"file_name":"...","width":...,"height":...}],
    "annotations": [{"image_id":0,"category_id":0}]  # optional
  }

Usage:
    python deepfake_detector.py --dataset demo.json \
        --images_dir ./Images --output results.json \
        --workers 4 --mode buffered
"""

import os
import sys
import json
import time
import mmap
import fcntl
import struct
import signal
import logging
import argparse
import threading
import multiprocessing
from pathlib import Path
from typing import Optional
import numpy as np

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s/%(threadName)s] %(levelname)s %(message)s",
)
log = logging.getLogger("deepfake")


# ─────────────────────────────────────────────
# OS Component 1: Shared Memory Buffer (mmap)
# Rationale: mmap avoids redundant copies between processes.
#   read/write would require pickling large arrays through a pipe;
#   mmap lets worker processes access the same physical pages directly.
# Trade-off: mmap is fast for repeated access to the same region, but
#   initial page faults can be expensive vs a single sequential read.
# ─────────────────────────────────────────────
class SharedImageBuffer:
    """
    Allocates a POSIX-style anonymous mmap region sized for one image frame.
    Workers write raw pixel bytes; the main process reads without extra copy.
    """
    HEADER_FMT = "=III"          # image_id (uint), width (uint), height (uint)
    HEADER_SIZE = struct.calcsize(HEADER_FMT)

    def __init__(self, max_w: int = 1024, max_h: int = 1024, channels: int = 3):
        self.max_w = max_w
        self.max_h = max_h
        self.channels = channels
        payload_size = max_w * max_h * channels
        self.total_size = self.HEADER_SIZE + payload_size

        # MAP_SHARED | MAP_ANONYMOUS — pages shared across fork()
        self._buf = mmap.mmap(-1, self.total_size, mmap.MAP_SHARED,
                              mmap.PROT_READ | mmap.PROT_WRITE)
        log.info("mmap: allocated %d bytes of shared image buffer", self.total_size)

    def write(self, image_id: int, pixels: np.ndarray) -> None:
        h, w, c = pixels.shape
        if h > self.max_h or w > self.max_w:
            raise ValueError(f"Image {w}x{h} exceeds buffer {self.max_w}x{self.max_h}")
        self._buf.seek(0)
        self._buf.write(struct.pack(self.HEADER_FMT, image_id, w, h))
        self._buf.write(pixels.astype(np.uint8).tobytes())

    def read(self) -> tuple[int, np.ndarray]:
        self._buf.seek(0)
        image_id, w, h = struct.unpack(self.HEADER_FMT,
                                       self._buf.read(self.HEADER_SIZE))
        raw = self._buf.read(w * h * self.channels)
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, self.channels)
        return image_id, pixels

    def close(self) -> None:
        self._buf.close()
        log.info("mmap: shared buffer released")


# ─────────────────────────────────────────────
# OS Component 2: Semaphore-based Producer/Consumer
# Rationale: A semaphore gate allows N workers to run concurrently
#   without busy-waiting. sem_wait (acquire) blocks the caller; the OS
#   scheduler parks the thread until sem_post (release) signals it.
#   This is equivalent to pthread_sem_wait / sem_post at the C level.
# Trade-off: Semaphores add syscall overhead per image (~1–5 µs), but
#   prevent CPU spinning that wastes cores on I/O-bound workloads.
# ─────────────────────────────────────────────
class WorkerPool:
    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self._sem = threading.Semaphore(n_workers)   # limits concurrency
        self._results: dict = {}
        self._lock = threading.Lock()                # guards _results
        self._threads: list[threading.Thread] = []

    def submit(self, fn, *args) -> None:
        """sem_wait equivalent: blocks if n_workers slots are all taken."""
        self._sem.acquire()
        t = threading.Thread(target=self._run, args=(fn, args), daemon=True)
        self._threads.append(t)
        t.start()

    def _run(self, fn, args) -> None:
        try:
            image_id, result = fn(*args)
            with self._lock:
                self._results[image_id] = result
        finally:
            self._sem.release()   # sem_post: free one slot

    def join_all(self) -> dict:
        for t in self._threads:
            t.join()
        return self._results


# ─────────────────────────────────────────────
# OS Component 3: I/O Strategy
# Buffered I/O  – Python default (fopen / FILE* equivalent).
#   OS merges small writes; good for many small images.
# Direct I/O    – open(O_DIRECT equivalent via low-level os.open).
#   Bypasses page cache; lower latency for large sequential reads
#   when you know the data won't be re-read (avoids cache pollution).
# Trade-off: Buffered wins for repeated access to the same file
#   (cache hit); direct wins for large one-shot pipelines.
# ─────────────────────────────────────────────
def load_image_buffered(path: str) -> Optional[np.ndarray]:
    """Standard buffered read — OS page cache handles prefetch."""
    try:
        # Use PIL if available; fall back to raw bytes for demo
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB").resize((224, 224))
            return np.array(img)
        except ImportError:
            # Demo fallback: synthesise pixel data from file bytes
            with open(path, "rb") as f:
                data = f.read(224 * 224 * 3)
            arr = np.frombuffer(data.ljust(224 * 224 * 3, b'\x00'),
                                dtype=np.uint8).reshape(224, 224, 3)
            return arr
    except OSError as e:
        log.error("open(buffered) failed for %s: %s", path, e)
        return None


def load_image_direct(path: str) -> Optional[np.ndarray]:
    """
    Simulates O_DIRECT-style read: opens with os.open, reads in aligned
    chunks, bypasses Python's internal buffering layer.
    On Linux you would add os.O_DIRECT; here we omit it for portability
    but keep the raw read/seek pattern to demonstrate the syscall path.
    """
    CHUNK = 4096  # align to page size (getconf PAGESIZE)
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            chunks = []
            while True:
                chunk = os.read(fd, CHUNK)   # raw read(2) syscall
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(fd)                     # always close(2)
        data = b"".join(chunks)
        arr = np.frombuffer(data[:224 * 224 * 3].ljust(224 * 224 * 3, b'\x00'),
                            dtype=np.uint8).reshape(224, 224, 3)
        return arr
    except OSError as e:
        log.error("os.open(direct) failed for %s: %s [errno %d]",
                  path, e.strerror, e.errno)
        return None


# ─────────────────────────────────────────────
# OS Component 4: Child Process via fork/exec pattern
# Rationale: CPU-heavy feature extraction runs in a subprocess to get
#   true parallelism past the GIL. multiprocessing.Process uses
#   os.fork() under the hood; we wait() on it explicitly.
# ─────────────────────────────────────────────
def _feature_worker(image_id: int, pixels_bytes: bytes,
                    shape: tuple, result_queue) -> None:
    """
    Runs in the child process (post-fork).
    Extracts lightweight statistical features as proxy for a CNN encoder.
    """
    pixels = np.frombuffer(pixels_bytes, dtype=np.uint8).reshape(shape)
    features = {
        "mean_rgb": pixels.mean(axis=(0, 1)).tolist(),
        "std_rgb":  pixels.std(axis=(0, 1)).tolist(),
        "laplacian_var": float(_laplacian_variance(pixels)),
    }
    result_queue.put((image_id, features))


def _laplacian_variance(img: np.ndarray) -> float:
    """
    Sharpness proxy via discrete Laplacian. Deepfakes often show lower
    high-frequency energy in blended regions.
    """
    gray = img.mean(axis=2).astype(np.float32)
    kern = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # manual 2-D convolution (no scipy dependency)
    out = np.zeros_like(gray)
    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            out[i, j] = (gray[i-1:i+2, j-1:j+2] * kern).sum()
    return float(out.var())


def extract_features_subprocess(image_id: int,
                                pixels: np.ndarray) -> dict:
    """fork/exec wrapper — wait() for child to finish."""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_feature_worker,
        args=(image_id, pixels.tobytes(), pixels.shape, q),
        name=f"feature-{image_id}"
    )
    p.start()
    p.join()           # wait(2) equivalent — blocks until child exits
    if p.exitcode != 0:
        log.warning("Child process %d exited with code %d",
                    p.pid, p.exitcode)
        return {}
    return q.get()


# ─────────────────────────────────────────────
# Detection Logic (heuristic classifier)
# In production: replace feature_to_score with a CNN forward pass.
# ─────────────────────────────────────────────
def classify(features: dict) -> tuple[str, float]:
    """
    Heuristic: low Laplacian variance + abnormal colour channel spread
    → elevated fake probability.
    Returns (label, confidence_0_to_1).
    """
    if not features:
        return "unknown", 0.0
    lap = features.get("laplacian_var", 1.0)
    std  = features.get("std_rgb", [64, 64, 64])
    channel_spread = max(std) - min(std)

    # Normalise scores (empirical thresholds — tune on real data)
    lap_score   = min(lap / 2000.0, 1.0)          # higher → more real
    color_score = min(channel_spread / 30.0, 1.0)  # imbalanced → more fake

    fake_prob = 0.5 * (1 - lap_score) + 0.5 * color_score
    label = "Fake" if fake_prob > 0.5 else "Real"
    return label, round(fake_prob, 4)


# ─────────────────────────────────────────────
# OS Component 5: fsync-safe result writer
# Rationale: Without fsync, write() may sit in the kernel's page cache
#   and be lost on crash. fsync(2) forces dirty pages to storage.
# Trade-off: fsync after every record is safest but slow. Batching N
#   records then fsyncing amortises the cost (here: fsync per batch).
# ─────────────────────────────────────────────
class SafeResultWriter:
    def __init__(self, path: str, batch_size: int = 10):
        self.path = path
        self.batch_size = batch_size
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        self._buf: list[str] = []
        self._write_raw('[\n')

    def write_result(self, record: dict) -> None:
        line = json.dumps(record, indent=2)
        self._buf.append(line)
        if len(self._buf) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        payload = ",\n".join(self._buf) + ",\n"
        self._write_raw(payload)
        os.fsync(self._fd)   # fsync(2): flush dirty pages to disk
        log.debug("fsync: flushed %d records to %s", len(self._buf), self.path)
        self._buf.clear()

    def _write_raw(self, text: str) -> None:
        data = text.encode()
        total = len(data)
        written = 0
        while written < total:
            n = os.write(self._fd, data[written:])   # write(2) syscall
            if n < 0:
                raise OSError("write() returned negative")
            written += n

    def close(self) -> None:
        self._flush()
        self._write_raw('null\n]\n')  # close JSON array cleanly
        os.fsync(self._fd)
        os.close(self._fd)            # close(2)
        log.info("Result file closed and synced: %s", self.path)


# ─────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────
def process_one(image_meta: dict, images_dir: str,
                io_mode: str, shared_buf: SharedImageBuffer) -> tuple[int, dict]:
    """
    Per-image pipeline:
      1. Load pixels (buffered or direct I/O)
      2. Write to shared mmap buffer
      3. Fork child to extract features
      4. Classify and return result
    """
    image_id = image_meta["id"]
    fname = image_meta["file_name"]
    path = os.path.join(images_dir, os.path.basename(fname))

    t0 = time.monotonic()

    # Step 1: Load image
    if io_mode == "direct":
        pixels = load_image_direct(path)
    else:
        pixels = load_image_buffered(path)

    if pixels is None:
        # File missing — synthesise noise for demo
        pixels = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        log.warning("Image %s not found — using synthetic pixels", path)

    # Step 2: Write to shared mmap
    shared_buf.write(image_id, pixels)

    # Step 3: Extract features in child process
    features = extract_features_subprocess(image_id, pixels)

    # Step 4: Classify
    label, confidence = classify(features)

    elapsed = time.monotonic() - t0
    result = {
        "image_id": image_id,
        "file_name": fname,
        "prediction": label,
        "fake_probability": confidence,
        "features": features,
        "io_mode": io_mode,
        "elapsed_ms": round(elapsed * 1000, 2),
    }
    log.info("[%s] image_id=%d  %-4s  conf=%.3f  %.1fms",
             io_mode, image_id, label, confidence, elapsed * 1000)
    return image_id, result


def run(dataset_path: str, images_dir: str, output_path: str,
        n_workers: int, io_mode: str) -> None:

    # ── Load dataset ──
    with open(dataset_path) as f:
        dataset = json.load(f)

    images    = dataset.get("images", [])
    cats      = {c["id"]: c["name"] for c in dataset.get("categories", [])}
    log.info("Dataset: %d images, categories: %s", len(images), cats)

    # ── Shared memory ──
    max_w = max((im.get("width",  1024) for im in images), default=1024)
    max_h = max((im.get("height", 1024) for im in images), default=1024)
    shared_buf = SharedImageBuffer(max_w=min(max_w, 1024),
                                   max_h=min(max_h, 1024))

    # ── Worker pool (semaphore-gated) ──
    pool = WorkerPool(n_workers)

    # ── Result writer (fsync-safe) ──
    writer = SafeResultWriter(output_path)

    # ── Dispatch ──
    t_start = time.monotonic()
    for meta in images:
        pool.submit(process_one, meta, images_dir, io_mode, shared_buf)

    results = pool.join_all()
    for _, rec in sorted(results.items()):
        writer.write_result(rec)

    writer.close()
    shared_buf.close()

    elapsed = time.monotonic() - t_start
    fake_count = sum(1 for r in results.values() if r["prediction"] == "Fake")
    log.info("Done: %d images in %.2fs  |  Fake=%d  Real=%d",
             len(results), elapsed, fake_count, len(results) - fake_count)
    print(f"\n✓ Results written to: {output_path}")
    print(f"  Total: {len(results)} images  |  Fake: {fake_count}  "
          f"Real: {len(results) - fake_count}  |  Time: {elapsed:.2f}s")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Deepfake Image Detector")
    p.add_argument("--dataset",  required=True, help="Path to COCO-style JSON")
    p.add_argument("--images_dir", default=".",  help="Root dir for image files")
    p.add_argument("--output",   default="results.json", help="Output JSON path")
    p.add_argument("--workers",  type=int, default=4, help="Max concurrent workers")
    p.add_argument("--mode",     choices=["buffered", "direct"], default="buffered",
                   help="I/O strategy: buffered (page cache) or direct (raw syscall)")
    return p.parse_args()


if __name__ == "__main__":
    # Graceful shutdown on SIGINT (Ctrl-C)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    args = parse_args()
    run(
        dataset_path=args.dataset,
        images_dir=args.images_dir,
        output_path=args.output,
        n_workers=args.workers,
        io_mode=args.mode,
    )
