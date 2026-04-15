# Deepfake Detection System — Design Document

## Overview

A production-grade deepfake image detection pipeline that satisfies the OS systems
programming requirements: scheduling, memory management, multiprocessing,
synchronisation, and correct system-call usage.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Main Process                       │
│                                                      │
│  JSON Dataset ──► WorkerPool (Semaphore-gated)       │
│                        │                             │
│              ┌─────────┴──────────┐                 │
│              │  Thread 1..N       │                 │
│              │  (sem_acquire)     │                 │
│              │     │              │                 │
│              │  load_image()      │  buffered I/O   │
│              │  (read / os.read)  │  or direct I/O  │
│              │     │              │                 │
│              │  SharedImageBuffer │  ◄── mmap()     │
│              │     │              │                 │
│              │  fork() ──► Child  │                 │
│              │  wait()            │                 │
│              │     │              │                 │
│              │  classify()        │                 │
│              │  (sem_release)     │                 │
│              └─────────┬──────────┘                 │
│                        │                             │
│              SafeResultWriter                        │
│              (write + fsync per batch)               │
└─────────────────────────────────────────────────────┘
```

---

## OS Components Implemented

### 1. Memory Management — `mmap`
- **What**: `mmap.mmap(-1, size, MAP_SHARED, PROT_READ|PROT_WRITE)` allocates
  an anonymous shared-memory region backed by the kernel's virtual memory system.
- **Why**: Avoids pickling/copying large pixel arrays through pipes.
  Worker threads write raw `uint8` bytes into the buffer; the classifier reads
  directly from the same physical pages — zero extra copy.
- **Trade-off**: mmap excels at repeated access (page stays warm in TLB).
  For single-pass, large sequential reads, `read(2)` with `O_SEQUENTIAL` hint
  may outperform due to kernel read-ahead prefetch.

### 2. Multiprocessing — `fork` / `wait`
- **What**: `multiprocessing.Process` uses `os.fork()` internally.
  The child runs `_feature_worker()` and returns via a `Queue`.
  The parent calls `p.join()` (equivalent to `wait(2)`/`waitpid(2)`).
- **Why**: Python's GIL prevents true CPU parallelism in threads.
  Forking gives each worker its own address space and Python interpreter,
  so Laplacian convolution runs in true parallel.
- **Error handling**: `p.exitcode` is checked post-join; non-zero exit is
  logged and an empty feature dict is returned instead of crashing the pipeline.

### 3. Synchronisation — Semaphore (`sem_wait` / `sem_post`)
- **What**: `threading.Semaphore(n_workers)` limits concurrent threads.
  `sem.acquire()` ≡ `sem_wait(3)` — blocks caller; OS scheduler parks thread.
  `sem.release()` ≡ `sem_post(3)` — wakes one blocked thread.
- **Why**: Without a semaphore, submitting 10,000 images would spawn 10,000
  threads simultaneously, exhausting file descriptors and stack memory.
  A mutex (`pthread_mutex_lock`) would serialise entirely; a semaphore allows
  controlled concurrency.
- **Race condition protection**: `_results` dict is protected by a separate
  `threading.Lock` (≡ `pthread_mutex_lock`) to prevent torn writes from
  concurrent thread completion.

### 4. File I/O — `open` / `read` / `write` / `close` / `fsync`

#### Buffered mode (`open` / Python file object)
- Uses `fopen(3)` / glibc stdio — OS page cache absorbs bursts of small reads.
- Optimal when the same image file is accessed more than once (cache hit).

#### Direct mode (`os.open` / `os.read`)
- Uses raw `open(2)` + `read(2)` syscalls without stdio buffering.
- On Linux you would add `O_DIRECT` to bypass the page cache entirely,
  useful for one-shot large pipelines where cached data would evict more
  valuable working-set pages.
- **Trade-off**: Direct I/O requires 512-byte-aligned buffers and is ~2×
  slower for small files; it saves memory pressure for TB-scale datasets.

#### Result writer (`write` + `fsync`)
- Results are written via `os.write(fd, data)` — raw `write(2)`.
- `os.fsync(fd)` is called every `batch_size` records, forcing dirty pages
  from the kernel's page cache to stable storage.
- **Trade-off**: `fsync` per record = maximum durability, minimum throughput.
  `fsync` per batch of 10 = ~10× throughput improvement with acceptable
  crash window (at most 10 records lost on power failure).
  `O_SYNC` on `open` would fsync every `write` automatically — safest but
  slowest, not used here because the batch approach is a reasonable middle ground.

---

## System Calls Used

| Syscall | Where | Purpose |
|---------|-------|---------|
| `mmap(2)` | `SharedImageBuffer.__init__` | Allocate shared memory |
| `fork(2)` | `multiprocessing.Process.start()` | Spawn feature extraction child |
| `wait(2)` / `waitpid(2)` | `p.join()` | Reap child, retrieve exit code |
| `open(2)` | `os.open()` in direct mode | Open image file, raw descriptor |
| `read(2)` | `os.read()` in direct mode | Read image bytes |
| `write(2)` | `os.write()` in `SafeResultWriter` | Write JSON results |
| `fsync(2)` | `os.fsync()` | Flush dirty pages to disk |
| `close(2)` | `os.close()` | Release file descriptor |
| `sem_wait` | `threading.Semaphore.acquire()` | Block thread when pool full |
| `sem_post` | `threading.Semaphore.release()` | Wake waiting thread |
| `pthread_mutex_lock` | `threading.Lock` | Protect shared `_results` dict |
| `signal(2)` | `signal.signal(SIGINT,...)` | Graceful Ctrl-C shutdown |

---

## Performance Trade-offs Summary

| Choice | Benefit | Cost |
|--------|---------|------|
| `mmap` vs `read/write` | Zero-copy pixel sharing | Initial page-fault overhead |
| Buffered vs Direct I/O | Cache reuse for small files | Cache pollution for large datasets |
| `fsync` per batch (10) | 10× write throughput | ≤10 records lost on crash |
| Semaphore (N workers) | Bounded resource usage | Semaphore syscall overhead ~1µs/op |
| `fork` for features | True CPU parallelism past GIL | Fork overhead ~1–5ms per image |

---

## Usage

```bash
# Install optional dependency for real images
pip install pillow numpy

# Run on the demo dataset (images don't need to exist — synthetic fallback used)
python deepfake_detector.py \
    --dataset demo.json \
    --images_dir ./Images/Test-Dev \
    --output results.json \
    --workers 4 \
    --mode buffered

# Try direct I/O mode
python deepfake_detector.py --dataset demo.json --mode direct
```

## Output Format

```json
[
  {
    "image_id": 0,
    "file_name": "Images/Test-Dev/e8ef73c50c.jpg",
    "prediction": "Real",
    "fake_probability": 0.312,
    "features": {
      "mean_rgb": [127.4, 118.2, 109.6],
      "std_rgb": [52.1, 48.3, 44.7],
      "laplacian_var": 1842.3
    },
    "io_mode": "buffered",
    "elapsed_ms": 14.2
  }
]
```

## Extending to a Real CNN

Replace `_feature_worker` with a torchvision or ONNX forward pass:

```python
import torch, torchvision
model = torchvision.models.efficientnet_b0(pretrained=True)
model.eval()
with torch.no_grad():
    tensor = preprocess(pixels)       # standard ImageNet normalisation
    logits = model(tensor.unsqueeze(0))
    fake_prob = torch.softmax(logits, 1)[0, 1].item()
```

The rest of the OS-level pipeline (mmap, semaphores, fsync) remains unchanged.
