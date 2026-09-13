# Kid Studio GPU Worker

GPU execution worker for Kid Studio video production.

Current target stack:

- LTX Video 13B 0.9.8 distilled BF16 for shot generation
- LatentSync 1.6 for dialogue lip-sync (next stage)
- FFmpeg/imageio for media output
- Optional RunPod Network Volume mounted at `/runpod-volume`

## Design principles

1. The RunPod network volume is a cache, not a hard dependency.
2. Missing persistent storage must not permanently break generation.
3. Models are loaded explicitly and one heavyweight engine is kept resident at a time.
4. Model unload performs Python GC and CUDA cache cleanup.
5. Health/status endpoints never trigger model downloads or generation.
6. Kid Studio remains the authoritative orchestrator; this worker only executes prepared jobs.
7. Kid Studio supplies deterministic prompts and approved shot-reference media; LTX prompt enhancement is disabled so the worker does not rewrite the Director's intent.

## API

- `GET /health`
- `GET /storage/status`
- `GET /models/status`
- `POST /models/load`
- `POST /models/unload`
- `POST /video/generate`

`POST /models/load` with `{"model":"ltx"}` loads the LTX pipeline and keeps it resident. Multiple `/video/generate` calls reuse that pipeline until `/models/unload` is called.

`/video/generate` requires CUDA and accepts:

- prompt
- conditioning media paths and/or URLs
- conditioning start frames
- conditioning strengths
- width / height
- number of frames / frame rate
- seed
- negative prompt
- optional CPU offload

The endpoint is never invoked automatically by startup or health checks.

## LTX configuration

The worker pins the LTX-Video codebase at commit:

`5260738e171955b66c0827f9af7e84d68fd1d919`

Default pipeline config:

`configs/ltxv-13b-0.9.8-distilled.yaml`

This uses BF16 rather than the FP8/Q8 path so the default remains compatible with the RTX A6000 48 GB target. Prompt-enhancer models are deliberately disabled because Kid Studio already creates deterministic production prompts.

## Storage behavior

The worker prefers `/runpod-volume` when it exists and is writable. If it is missing, the worker falls back to `/workspace/kid-studio-cache` instead of failing startup.

Model files are stored under:

`<selected-cache-root>/models/<model>/<version>/`

The recovery utility and live model loader use the same directory, preventing duplicate copies of large model weights.

## Model recovery

Pinned models:

- LTX: `Lightricks/LTX-Video`, `ltxv-13b-0.9.8-distilled.safetensors` plus the 0.9.8 spatial upscaler
- LatentSync: `ByteDance/LatentSync-1.6`

Restore a model into the currently selected cache root:

```bash
python3 bootstrap_models.py ltx
```

or:

```bash
python3 bootstrap_models.py latentsync
```

## Environment

- `RUNPOD_VOLUME_PATH` default: `/runpod-volume`
- `LOCAL_CACHE_PATH` default: `/workspace/kid-studio-cache`
- `LTX_ROOT` default: `/opt/LTX-Video`
- `LTX_CONFIG` default: `/opt/LTX-Video/configs/ltxv-13b-0.9.8-distilled.yaml`
- `KID_STUDIO_OUTPUT_PATH` default: `/workspace/outputs`

When the network volume exists and is writable it is preferred. Otherwise the worker automatically falls back to local ephemeral storage.
