import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional

import imageio
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download

from ltx_video.inference import (
    calculate_padding,
    create_latent_upsampler,
    create_ltx_video_pipeline,
    get_device,
    get_total_gpu_memory,
    get_unique_filename,
    load_pipeline_config,
    prepare_conditioning,
    seed_everething,
)
from ltx_video.pipelines.pipeline_ltx_video import LTXMultiScalePipeline
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

from model_manifest import MODEL_MANIFESTS
from storage import select_cache_root


class LTXEngine:
    """Resident LTX 13B distilled BF16 engine for Kid Studio shots."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.ltx_root = Path(os.getenv("LTX_ROOT", "/opt/LTX-Video"))
        self.config_path = Path(
            os.getenv(
                "LTX_CONFIG",
                str(self.ltx_root / "configs" / "ltxv-13b-0.9.8-distilled.yaml"),
            )
        )
        if not self.config_path.exists():
            raise RuntimeError(f"LTX pipeline config not found: {self.config_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required to load LTX")

        self.device = get_device()
        self.pipeline_config = load_pipeline_config(str(self.config_path))
        manifest = MODEL_MANIFESTS["ltx"]
        cache_root, _ = select_cache_root()
        model_dir = cache_root / "models" / "ltx" / manifest.version
        model_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_name = self.pipeline_config["checkpoint_path"]
        checkpoint_local = model_dir / checkpoint_name
        if checkpoint_local.exists():
            checkpoint_path = str(checkpoint_local)
        else:
            checkpoint_path = hf_hub_download(
                repo_id="Lightricks/LTX-Video",
                filename=checkpoint_name,
                repo_type="model",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
            )

        upscaler_name = self.pipeline_config.get("spatial_upscaler_model_path")
        upscaler_local = model_dir / upscaler_name if upscaler_name else None
        if upscaler_local and upscaler_local.exists():
            upscaler_path = str(upscaler_local)
        elif upscaler_name:
            upscaler_path = hf_hub_download(
                repo_id="Lightricks/LTX-Video",
                filename=upscaler_name,
                repo_type="model",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
            )
        else:
            upscaler_path = None

        precision = self.pipeline_config["precision"]
        text_encoder = self.pipeline_config["text_encoder_model_name_or_path"]
        sampler = self.pipeline_config.get("sampler")

        # Kid Studio already produces detailed deterministic prompts. Disable
        # LTX prompt-enhancer models to save VRAM and prevent prompt rewriting.
        pipeline = create_ltx_video_pipeline(
            ckpt_path=checkpoint_path,
            precision=precision,
            text_encoder_model_name_or_path=text_encoder,
            sampler=sampler,
            device=self.device,
            enhance_prompt=False,
        )

        if self.pipeline_config.get("pipeline_type") == "multi-scale":
            if not upscaler_path:
                raise RuntimeError("LTX multi-scale config requires the spatial upscaler")
            latent_upsampler = create_latent_upsampler(upscaler_path, pipeline.device)
            pipeline = LTXMultiScalePipeline(
                pipeline,
                latent_upsampler=latent_upsampler,
            )

        self.pipeline = pipeline
        self.precision = precision
        self.model_dir = model_dir
        self.loaded = True

    @staticmethod
    def _download(url: str, destination: Path) -> Path:
        with requests.get(url, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        return destination

    def _resolve_conditioning_media(
        self,
        media_paths: Optional[List[str]],
        media_urls: Optional[List[str]],
        workdir: Path,
    ) -> List[str]:
        resolved: List[str] = []

        for item in media_paths or []:
            path = Path(item)
            if not path.exists():
                raise ValueError(f"Conditioning media does not exist: {item}")
            resolved.append(str(path))

        for index, url in enumerate(media_urls or []):
            suffix = Path(url.split("?", 1)[0]).suffix or ".png"
            destination = workdir / f"conditioning_{index}{suffix}"
            self._download(url, destination)
            resolved.append(str(destination))

        return resolved

    @staticmethod
    def _skip_strategy(stg_mode: str) -> SkipLayerStrategy:
        mode = str(stg_mode or "attention_values").lower()
        if mode in {"stg_av", "attention_values"}:
            return SkipLayerStrategy.AttentionValues
        if mode in {"stg_as", "attention_skip"}:
            return SkipLayerStrategy.AttentionSkip
        if mode in {"stg_r", "residual"}:
            return SkipLayerStrategy.Residual
        if mode in {"stg_t", "transformer_block"}:
            return SkipLayerStrategy.TransformerBlock
        raise ValueError(f"Invalid LTX spatiotemporal guidance mode: {stg_mode}")

    def generate(self, data: Dict[str, object]) -> Dict[str, object]:
        with self._lock:
            if not self.loaded or self.pipeline is None:
                raise RuntimeError("LTX engine is not loaded")

            prompt = str(data.get("prompt") or "").strip()
            if not prompt:
                raise ValueError("Missing required input: prompt")

            width = int(data.get("width") or 768)
            height = int(data.get("height") or 512)
            num_frames = int(data.get("num_frames") or 121)
            frame_rate = int(data.get("frame_rate") or 24)
            seed = int(data.get("seed") or 171198)
            negative_prompt = str(
                data.get("negative_prompt")
                or "worst quality, inconsistent motion, blurry, jittery, distorted"
            )

            if width <= 0 or height <= 0 or num_frames <= 0 or frame_rate <= 0:
                raise ValueError("width, height, num_frames and frame_rate must be positive")

            output_root = Path(
                str(
                    data.get("output_dir")
                    or os.getenv("KID_STUDIO_OUTPUT_PATH", "/workspace/outputs")
                )
            )
            output_root.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory(prefix="kid-studio-ltx-") as temp_name:
                temp_dir = Path(temp_name)
                conditioning = self._resolve_conditioning_media(
                    data.get("conditioning_media_paths") or [],
                    data.get("conditioning_media_urls") or [],
                    temp_dir,
                )

                start_frames = [
                    int(value) for value in (data.get("conditioning_start_frames") or [])
                ]
                strengths = [
                    float(value) for value in (data.get("conditioning_strengths") or [])
                ]

                if conditioning and not start_frames:
                    start_frames = [0] * len(conditioning)
                if conditioning and not strengths:
                    strengths = [1.0] * len(conditioning)
                if len(start_frames) != len(conditioning):
                    raise ValueError(
                        "conditioning_start_frames must match conditioning media count"
                    )
                if len(strengths) != len(conditioning):
                    raise ValueError(
                        "conditioning_strengths must match conditioning media count"
                    )
                if any(value < 0 or value >= num_frames for value in start_frames):
                    raise ValueError("conditioning_start_frames must fall inside the shot")
                if any(value < 0 or value > 1 for value in strengths):
                    raise ValueError("conditioning strengths must be between 0 and 1")

                height_padded = ((height - 1) // 32 + 1) * 32
                width_padded = ((width - 1) // 32 + 1) * 32
                num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1
                padding = calculate_padding(
                    height,
                    width,
                    height_padded,
                    width_padded,
                )

                seed_everething(seed)
                offload_requested = bool(data.get("offload_to_cpu", False))
                offload_to_cpu = offload_requested and get_total_gpu_memory() < 30

                conditioning_items = (
                    prepare_conditioning(
                        conditioning_media_paths=conditioning,
                        conditioning_strengths=strengths,
                        conditioning_start_frames=start_frames,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        padding=padding,
                        pipeline=self.pipeline,
                    )
                    if conditioning
                    else None
                )

                call_config = dict(self.pipeline_config)
                stg_mode = call_config.pop("stg_mode", "attention_values")
                skip_layer_strategy = self._skip_strategy(stg_mode)

                generator = torch.Generator(device=self.device).manual_seed(seed)
                images = self.pipeline(
                    **call_config,
                    skip_layer_strategy=skip_layer_strategy,
                    generator=generator,
                    output_type="pt",
                    callback_on_step_end=None,
                    height=height_padded,
                    width=width_padded,
                    num_frames=num_frames_padded,
                    frame_rate=frame_rate,
                    prompt=prompt,
                    prompt_attention_mask=None,
                    negative_prompt=negative_prompt,
                    negative_prompt_attention_mask=None,
                    media_items=None,
                    conditioning_items=conditioning_items,
                    is_video=True,
                    vae_per_channel_normalize=True,
                    image_cond_noise_scale=float(data.get("image_cond_noise_scale") or 0.15),
                    mixed_precision=(self.precision == "mixed_precision"),
                    offload_to_cpu=offload_to_cpu,
                    device=self.device,
                    enhance_prompt=False,
                ).images

                pad_left, pad_right, pad_top, pad_bottom = padding
                crop_bottom = -pad_bottom if pad_bottom else images.shape[3]
                crop_right = -pad_right if pad_right else images.shape[4]
                images = images[
                    :, :, :num_frames, pad_top:crop_bottom, pad_left:crop_right
                ]

                video_np = images[0].permute(1, 2, 3, 0).cpu().float().numpy()
                video_np = (video_np * 255).astype(np.uint8)
                output_path = get_unique_filename(
                    "video_output_0",
                    ".mp4",
                    prompt=prompt,
                    seed=seed,
                    resolution=(height, width, num_frames),
                    dir=output_root,
                )

                with imageio.get_writer(output_path, fps=frame_rate) as video:
                    for frame in video_np:
                        video.append_data(frame)

                del images
                del video_np
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return {
                    "ok": True,
                    "engine": "ltx",
                    "model": "LTX-Video 13B 0.9.8 distilled BF16",
                    "resident_pipeline": True,
                    "model_dir": str(self.model_dir),
                    "output_path": str(output_path),
                    "size_bytes": output_path.stat().st_size,
                    "width": width,
                    "height": height,
                    "num_frames": num_frames,
                    "frame_rate": frame_rate,
                    "seed": seed,
                    "conditioning_count": len(conditioning),
                }

    def close(self) -> None:
        with self._lock:
            self.loaded = False
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
