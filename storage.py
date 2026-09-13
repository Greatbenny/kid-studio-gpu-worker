import os
import shutil
from pathlib import Path
from typing import Dict, Tuple


DEFAULT_VOLUME_ROOT = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
DEFAULT_LOCAL_CACHE_ROOT = Path(os.getenv("LOCAL_CACHE_PATH", "/workspace/kid-studio-cache"))


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".kid_studio_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def select_cache_root() -> Tuple[Path, bool]:
    """Return (cache_root, persistent).

    The RunPod network volume is preferred when mounted and writable. If it is
    absent or unavailable, use local ephemeral storage so the worker can still
    start and recover models later.
    """
    if DEFAULT_VOLUME_ROOT.exists() and _is_writable_directory(DEFAULT_VOLUME_ROOT):
        return DEFAULT_VOLUME_ROOT, True

    DEFAULT_LOCAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return DEFAULT_LOCAL_CACHE_ROOT, False


def configure_cache_environment() -> Dict[str, str]:
    root, persistent = select_cache_root()

    hf_home = root / "huggingface"
    hf_hub = hf_home / "hub"
    torch_home = root / "torch"
    tmpdir = root / "tmp"
    models = root / "models"

    for path in (hf_home, hf_hub, torch_home, tmpdir, models):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_hub))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_hub))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_hub))
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    os.environ.setdefault("TMPDIR", str(tmpdir))

    return {
        "root": str(root),
        "persistent": str(persistent).lower(),
        "hf_home": str(hf_home),
        "hf_hub_cache": str(hf_hub),
        "torch_home": str(torch_home),
        "tmpdir": str(tmpdir),
        "models_root": str(models),
    }


def storage_status() -> Dict[str, object]:
    root, persistent = select_cache_root()
    available = root.exists()

    result: Dict[str, object] = {
        "available": available,
        "persistent": persistent,
        "root": str(root),
        "configured_volume_root": str(DEFAULT_VOLUME_ROOT),
        "using_fallback": not persistent,
    }

    if available:
        try:
            usage = shutil.disk_usage(root)
            result.update(
                {
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                    "free_bytes": usage.free,
                }
            )
        except Exception as exc:
            result["usage_error"] = str(exc)

    result.update(
        {
            "hf_home": os.getenv("HF_HOME", ""),
            "hf_hub_cache": os.getenv("HF_HUB_CACHE", ""),
            "torch_home": os.getenv("TORCH_HOME", ""),
            "tmpdir": os.getenv("TMPDIR", ""),
        }
    )
    return result
