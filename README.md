# Kid Studio GPU Worker

GPU execution worker for Kid Studio video production.

Initial target stack:

- LTX Video for shot generation
- LatentSync for dialogue lip-sync
- FFmpeg for media utilities
- Optional RunPod Network Volume mounted at `/runpod-volume`

## Design principles

1. The RunPod network volume is a cache, not a hard dependency.
2. Missing persistent storage must not crash the worker.
3. Models are loaded explicitly and one heavyweight engine is kept resident at a time.
4. Model unload performs Python GC and CUDA cache cleanup.
5. No paid generation is triggered by health/status endpoints.
6. Kid Studio remains the authoritative orchestrator; this worker only executes prepared jobs.

## Initial API

- `GET /health`
- `GET /storage/status`
- `GET /models/status`
- `POST /models/load`
- `POST /models/unload`

The first implementation is intentionally non-generative. LTX and LatentSync execution endpoints will be added after the worker lifecycle and persistent-storage behavior are verified on a GPU Pod.

## Storage behavior

The worker prefers the RunPod network volume when `/runpod-volume` exists and is writable. If it is missing, the worker falls back to `/workspace/kid-studio-cache` instead of failing startup.

This makes the network volume a cache rather than a hard dependency. A missing or expired volume can be rebuilt from the pinned model manifest.

## Model recovery

Pinned models:

- LTX: `Lightricks/LTX-Video`, `ltxv-13b-0.9.8-distilled-fp8`
- LatentSync: `ByteDance/LatentSync-1.6`

Restore a model into the currently selected cache root:

```bash
python3 bootstrap_models.py ltx
```

or:

```bash
python3 bootstrap_models.py latentsync
```

Where known, SHA256 checksums are verified after download.

## Environment

- `RUNPOD_VOLUME_PATH` default: `/runpod-volume`
- `LOCAL_CACHE_PATH` default: `/workspace/kid-studio-cache`
- `HF_HOME` defaults to the selected writable cache root
- `TORCH_HOME` defaults to the selected writable cache root

When the network volume exists and is writable it is preferred. Otherwise the worker automatically falls back to local ephemeral storage.
