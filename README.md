# Kid Studio GPU Worker

GPU execution worker for Kid Studio video production.

Current target stack:

- LTX Video 13B 0.9.8 distilled BF16 for shot generation
- LatentSync 1.6 (512px) for dialogue lip-sync
- FFmpeg/imageio for media output
- Optional RunPod Network Volume mounted at `/runpod-volume`

## Design principles

1. The RunPod network volume is a cache, not a hard dependency.
2. Missing persistent storage must not permanently break generation.
3. Models are loaded explicitly and only one heavyweight engine is kept resident at a time.
4. Model unload performs Python GC and CUDA cache cleanup.
5. Health/status endpoints never trigger model downloads or generation.
6. Kid Studio remains the authoritative orchestrator; this worker only executes prepared jobs.
7. Kid Studio supplies deterministic prompts and approved shot-reference media; LTX prompt enhancement is disabled so the worker does not rewrite the Director's intent.
8. LatentSync receives the already-approved dialogue audio; it never regenerates TTS.

## API

- `GET /health`
- `GET /storage/status`
- `GET /models/status`
- `POST /models/load`
- `POST /models/unload`
- `POST /video/generate`
- `POST /lipsync`

`POST /models/load` with `{"model":"ltx"}` loads LTX and keeps it resident. Multiple `/video/generate` calls reuse that pipeline until another model is requested or `/models/unload` is called.

`POST /models/load` with `{"model":"latentsync"}` unloads LTX first, loads LatentSync once, and keeps it resident for repeated `/lipsync` calls.

`/video/generate` requires CUDA and accepts deterministic shot prompts plus conditioning media, target frames/strengths, resolution, frame count/FPS and seed.

`/lipsync` requires CUDA and accepts:

- `video`: local path or HTTP(S) URL for the generated raw shot
- `audio`: local path or HTTP(S) URL for the existing approved dialogue audio
- `guidance_scale`: 1.0-3.0, default 1.5
- `inference_steps`: 20-50, default 20
- `seed`: default 1247
- optional output directory

Startup and health checks do not load models or run inference.

## LTX configuration

The worker pins the LTX-Video codebase at commit:

`5260738e171955b66c0827f9af7e84d68fd1d919`

Default pipeline config:

`configs/ltxv-13b-0.9.8-distilled.yaml`

This uses BF16 rather than the FP8/Q8 path so the default remains compatible with the RTX A6000 48 GB target. Prompt-enhancer models are deliberately disabled because Kid Studio already creates deterministic production prompts.

## LatentSync configuration

The worker pins ByteDance LatentSync at commit:

`a229c3948406bc2cf6eaf4873e662e70c6a04746`

Default inference config:

`configs/unet/stage2_512.yaml`

The worker follows the official LatentSync 1.6 inference defaults: 512px processing, 20 steps, guidance 1.5, and DeepCache enabled. The UNet, Whisper audio encoder, VAE and lip-sync pipeline stay resident across dialogue shots.

LatentSync 1.6 requires substantially less VRAM than the LTX 13B stage, so the same 48 GB GPU session can unload LTX and load LatentSync afterward.

## Storage behavior

The worker prefers `/runpod-volume` when it exists and is writable. If it is missing, the worker falls back to `/workspace/kid-studio-cache` instead of failing startup.

Model files are stored under:

`<selected-cache-root>/models/<model>/<version>/`

The recovery utility and live model loader use the same directory, preventing duplicate copies of large model weights.

## Model recovery

Pinned models:

- LTX: `Lightricks/LTX-Video`, `ltxv-13b-0.9.8-distilled.safetensors` plus the 0.9.8 spatial upscaler
- LatentSync: `ByteDance/LatentSync-1.6`, `latentsync_unet.pt` plus `whisper/tiny.pt`

Restore a model into the currently selected cache root:

```bash
python3 bootstrap_models.py ltx
```

or:

```bash
python3 bootstrap_models.py latentsync
```

## Intended production lifecycle

```text
start GPU pod
  -> health check
  -> load LTX once
  -> generate all queued visual shots
  -> unload LTX
  -> load LatentSync once
  -> lip-sync only dialogue shots using existing TTS
  -> unload LatentSync
  -> persist/upload completed media
  -> terminate GPU pod
```

## Environment

- `RUNPOD_VOLUME_PATH` default: `/runpod-volume`
- `LOCAL_CACHE_PATH` default: `/workspace/kid-studio-cache`
- `LTX_ROOT` default: `/opt/LTX-Video`
- `LTX_CONFIG` default: `/opt/LTX-Video/configs/ltxv-13b-0.9.8-distilled.yaml`
- `LATENTSYNC_ROOT` default: `/opt/LatentSync`
- `LATENTSYNC_CONFIG` default: `/opt/LatentSync/configs/unet/stage2_512.yaml`
- `KID_STUDIO_OUTPUT_PATH` default: `/workspace/outputs`

When the network volume exists and is writable it is preferred. Otherwise the worker automatically falls back to local ephemeral storage.
