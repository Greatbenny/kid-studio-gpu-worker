import gc
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    purpose: str
    implemented: bool = False


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "ltx": ModelSpec(
        key="ltx",
        family="LTX-Video",
        purpose="video_generation",
        implemented=False,
    ),
    "latentsync": ModelSpec(
        key="latentsync",
        family="LatentSync",
        purpose="lip_sync",
        implemented=False,
    ),
}


class ModelManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._engine: Optional[Any] = None
        self._model_key: Optional[str] = None

    @property
    def model_key(self) -> Optional[str]:
        return self._model_key

    def status(self) -> Dict[str, object]:
        with self._lock:
            return {
                "loaded_model": self._model_key,
                "loaded": self._engine is not None,
                "registry": {
                    key: {
                        "family": spec.family,
                        "purpose": spec.purpose,
                        "implemented": spec.implemented,
                    }
                    for key, spec in MODEL_REGISTRY.items()
                },
            }

    def load(self, model_key: str) -> Dict[str, object]:
        key = str(model_key or "").strip().lower()
        if key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {key}")

        spec = MODEL_REGISTRY[key]
        if not spec.implemented:
            return {
                "ok": False,
                "loaded": False,
                "model": key,
                "message": (
                    f"{spec.family} is registered but inference loading is not enabled yet. "
                    "This foundation build performs no paid/generative work."
                ),
            }

        with self._lock:
            if self._model_key == key and self._engine is not None:
                return {"ok": True, "loaded": True, "model": key, "reused": True}

            self.unload()
            raise NotImplementedError(f"Model loader not implemented: {key}")

    def unload(self) -> Dict[str, object]:
        with self._lock:
            previous = self._model_key

            if self._engine is not None:
                close = getattr(self._engine, "close", None)
                if callable(close):
                    close()

            self._engine = None
            self._model_key = None
            gc.collect()

            cuda_cleared = False
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    cuda_cleared = True
            except Exception:
                pass

            return {
                "ok": True,
                "previous_model": previous,
                "loaded_model": None,
                "cuda_cache_cleared": cuda_cleared,
            }


model_manager = ModelManager()
