import os
import sys
import tempfile
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse

import requests

from bootstrap_models import ensure_model
from storage import configure_cache_environment


class LatentSyncEngine:
    """Resident LatentSync 1.6 pipeline for repeated dialogue-shot lip-sync."""

    def __init__(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for LatentSync 1.6")

        self.torch = torch
        self.root = Path(os.getenv("LATENTSYNC_ROOT", "/opt/LatentSync"))
        self.config_path = Path(
            os.getenv(
                "LATENTSYNC_CONFIG",
                str(self.root / "configs/unet/stage2_512.yaml"),
            )
        )
        if not self.root.exists() or not self.config_path.exists():
            raise RuntimeError("Pinned LatentSync source/config is missing from the worker image")

        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))

        restored = ensure_model("latentsync", verify=True)
        artifact_paths = {
            item["filename"]: Path(item["path"]) for item in restored["artifacts"]
        }
        self.ckpt_path = artifact_paths["latentsync_unet.pt"]
        self.whisper_path = artifact_paths["whisper/tiny.pt"]

        from accelerate.utils import set_seed
        from DeepCache import DeepCacheSDHelper
        from diffusers import AutoencoderKL, DDIMScheduler
        from omegaconf import OmegaConf
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from latentsync.whisper.audio2feature import Audio2Feature

        self.set_seed = set_seed
        self.config = OmegaConf.load(str(self.config_path))
        self.dtype = (
            torch.float16
            if torch.cuda.get_device_capability()[0] > 7
            else torch.float32
        )

        scheduler = DDIMScheduler.from_pretrained(str(self.root / "configs"))
        self.audio_encoder = Audio2Feature(
            model_path=str(self.whisper_path),
            device="cuda",
            num_frames=self.config.data.num_frames,
            audio_feat_length=self.config.data.audio_feat_length,
        )

        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=self.dtype,
        )
        self.vae.config.scaling_factor = 0.18215
        self.vae.config.shift_factor = 0

        self.unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(self.config.model),
            str(self.ckpt_path),
            device="cpu",
        )
        self.unet = self.unet.to(dtype=self.dtype)

        self.pipeline = LipsyncPipeline(
            vae=self.vae,
            audio_encoder=self.audio_encoder,
            unet=self.unet,
            scheduler=scheduler,
        ).to("cuda")

        self.deepcache = DeepCacheSDHelper(pipe=self.pipeline)
        self.deepcache.set_params(cache_interval=3, cache_branch_id=0)
        self.deepcache.enable()

        env = configure_cache_environment()
        self.work_root = Path(env["root"]) / "temp" / "latentsync"
        self.work_root.mkdir(parents=True, exist_ok=True)

    def _materialize_input(self, value: str, label: str, work_dir: Path) -> Path:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError(f"Missing required input: {label}")

        if raw.startswith("http://") or raw.startswith("https://"):
            suffix = Path(urlparse(raw).path).suffix or (".mp4" if label == "video" else ".wav")
            target = work_dir / f"input_{label}{suffix}"
            with requests.get(raw, stream=True, timeout=(20, 600)) as response:
                response.raise_for_status()
                with target.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            return target

        path = Path(raw)
        if not path.exists() or not path.is_file():
            raise ValueError(f"{label.capitalize()} path does not exist: {path}")
        return path

    def sync(self, data: Dict[str, object]) -> Dict[str, object]:
        guidance_scale = float(data.get("guidance_scale", 1.5))
        inference_steps = int(data.get("inference_steps", 20))
        seed = int(data.get("seed", 1247))

        if not 1.0 <= guidance_scale <= 3.0:
            raise ValueError("guidance_scale must be between 1.0 and 3.0")
        if not 20 <= inference_steps <= 50:
            raise ValueError("inference_steps must be between 20 and 50")

        with tempfile.TemporaryDirectory(prefix="job-", dir=str(self.work_root)) as tmp:
            work_dir = Path(tmp)
            video = self._materialize_input(str(data.get("video") or ""), "video", work_dir)
            audio = self._materialize_input(str(data.get("audio") or ""), "audio", work_dir)

            output_dir = Path(str(data.get("output_dir") or self.work_root / "outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"latentsync-{seed}-{os.urandom(6).hex()}.mp4"
            temp_dir = work_dir / "pipeline-temp"
            temp_dir.mkdir(parents=True, exist_ok=True)

            if seed >= 0:
                self.set_seed(seed)
            else:
                self.torch.seed()

            self.pipeline(
                video_path=str(video),
                audio_path=str(audio),
                video_out_path=str(output_path),
                num_frames=self.config.data.num_frames,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                weight_dtype=self.dtype,
                width=self.config.data.resolution,
                height=self.config.data.resolution,
                mask_image_path=str(self.root / self.config.data.mask_image_path),
                temp_dir=str(temp_dir),
            )

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError("LatentSync completed without producing an output video")

            return {
                "ok": True,
                "engine": "latentsync-1.6",
                "resolution": int(self.config.data.resolution),
                "output_path": str(output_path),
                "size_bytes": output_path.stat().st_size,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "inference_steps": inference_steps,
                "deepcache": True,
                "input_audio_preserved": True,
            }

    def close(self) -> None:
        try:
            if hasattr(self, "deepcache"):
                self.deepcache.disable()
        except Exception:
            pass

        for name in ("deepcache", "pipeline", "unet", "vae", "audio_encoder"):
            if hasattr(self, name):
                try:
                    delattr(self, name)
                except Exception:
                    pass
