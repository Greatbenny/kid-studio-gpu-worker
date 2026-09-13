import platform
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_manager import model_manager
from storage import configure_cache_environment, storage_status


CACHE_ENV = configure_cache_environment()
WORKER_BUILD = "gpu-ltx-latentsync-v3"

app = FastAPI(title="Kid Studio GPU Worker", version=WORKER_BUILD)


class ModelLoadRequest(BaseModel):
    model: str


class VideoGenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    conditioning_media_paths: List[str] = Field(default_factory=list)
    conditioning_media_urls: List[str] = Field(default_factory=list)
    conditioning_start_frames: List[int] = Field(default_factory=list)
    conditioning_strengths: List[float] = Field(default_factory=list)
    width: int = 768
    height: int = 512
    num_frames: int = 121
    frame_rate: int = 24
    seed: int = 171198
    negative_prompt: Optional[str] = None
    offload_to_cpu: bool = False
    output_dir: Optional[str] = None
    timeout_seconds: int = 7200


class LipSyncRequest(BaseModel):
    video: str = Field(min_length=1, description="Local path or HTTP(S) URL to raw video")
    audio: str = Field(min_length=1, description="Local path or HTTP(S) URL to approved dialogue audio")
    guidance_scale: float = Field(default=1.5, ge=1.0, le=3.0)
    inference_steps: int = Field(default=20, ge=20, le=50)
    seed: int = 1247
    output_dir: Optional[str] = None


def gpu_status() -> Dict[str, object]:
    result: Dict[str, object] = {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }

    try:
        import torch

        available = torch.cuda.is_available()
        result["cuda_available"] = available
        result["device_count"] = torch.cuda.device_count() if available else 0

        devices = []
        if available:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                free_bytes = None
                total_bytes = None
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                except Exception:
                    total_bytes = int(props.total_memory)

                devices.append(
                    {
                        "index": index,
                        "name": props.name,
                        "compute_capability": f"{props.major}.{props.minor}",
                        "total_memory_bytes": int(total_bytes or props.total_memory),
                        "free_memory_bytes": int(free_bytes) if free_bytes is not None else None,
                    }
                )

        result["devices"] = devices
    except Exception as exc:
        result["error"] = str(exc)

    return result


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "ok": True,
        "service": "kid-studio-gpu-worker",
        "worker_build": WORKER_BUILD,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "storage": storage_status(),
        "gpu": gpu_status(),
        "models": model_manager.status(),
        "generation_enabled": True,
        "video_engine": "ltx",
        "lip_sync_enabled": True,
        "lip_sync_engine": "latentsync-1.6",
    }


@app.get("/storage/status")
def get_storage_status() -> Dict[str, object]:
    return storage_status()


@app.get("/models/status")
def get_models_status() -> Dict[str, object]:
    return model_manager.status()


@app.post("/models/load")
def load_model(request: ModelLoadRequest) -> Dict[str, object]:
    try:
        return model_manager.load(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/models/unload")
def unload_model() -> Dict[str, object]:
    return model_manager.unload()


@app.post("/video/generate")
def generate_video(request: VideoGenerateRequest) -> Dict[str, object]:
    gpu = gpu_status()
    if not gpu.get("cuda_available"):
        raise HTTPException(status_code=503, detail="CUDA GPU is required for LTX generation")

    try:
        engine = model_manager.require_engine("ltx")
        return engine.generate(request.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/lipsync")
def lip_sync(request: LipSyncRequest) -> Dict[str, object]:
    gpu = gpu_status()
    if not gpu.get("cuda_available"):
        raise HTTPException(status_code=503, detail="CUDA GPU is required for LatentSync")

    try:
        engine = model_manager.require_engine("latentsync")
        return engine.sync(request.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
def root() -> Dict[str, object]:
    return {
        "service": "kid-studio-gpu-worker",
        "worker_build": WORKER_BUILD,
        "health": "/health",
        "video_generate": "/video/generate",
        "lip_sync": "/lipsync",
    }
