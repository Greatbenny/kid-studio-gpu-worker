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

## Environment

- `RUNPOD_VOLUME_PATH` default: `/runpod-volume`
- `LOCAL_CACHE_PATH` default: `/workspace/kid-studio-cache`
- `HF_HOME` defaults to the selected writable cache root
- `TORCH_HOME` defaults to the selected writable cache root

When the network volume exists and is writable it is preferred. Otherwise the worker automatically falls back to local ephemeral storage.
