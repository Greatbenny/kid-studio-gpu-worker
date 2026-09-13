import os
import platform
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_manager import model_manager
from storage import configure_cache_environment, storage_status


CACHE_ENV = configure_cache_environment()
WORKER_BUILD = "gpu-foundation-v1"

app = FastAPI(title="Kid Studio GPU Worker", version=WORKER_BUILD)


class ModelLoadRequest(BaseModel):
    model: str


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
                free_bytes: Optional[int] = None
                total_bytes: Optional[int] = None
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
    storage = storage_status()
    gpu = gpu_status()

    return {
        "ok": True,
        "service": "kid-studio-gpu-worker",
        "worker_build": WORKER_BUILD,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "storage": storage,
        "gpu": gpu,
        "models": model_manager.status(),
        "safe_mode": True,
        "generation_enabled": False,
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


@app.get("/")
def root() -> Dict[str, object]:
    return {
        "service": "kid-studio-gpu-worker",
        "worker_build": WORKER_BUILD,
        "health": "/health",
    }
