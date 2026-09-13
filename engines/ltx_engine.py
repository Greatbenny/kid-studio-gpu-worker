import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import requests


class LTXEngine:
    """LTX 13B distilled video generation wrapper.

    The worker image contains a pinned checkout of the official LTX-Video
    repository. This wrapper validates that runtime, resolves remote
    conditioning assets to local files, invokes the official inference entry
    point, and returns the generated MP4 path.

    Generation is only performed when generate() is explicitly called.
    """

    def __init__(self) -> None:
        self.ltx_root = Path(os.getenv("LTX_ROOT", "/opt/LTX-Video"))
        self.config_path = Path(
            os.getenv(
                "LTX_CONFIG",
                str(self.ltx_root / "configs" / "ltxv-13b-0.9.8-distilled.yaml"),
            )
        )
        self.python = os.getenv("LTX_PYTHON", "python3")
        self.inference_script = self.ltx_root / "inference.py"

        if not self.inference_script.exists():
            raise RuntimeError(f"LTX inference script not found: {self.inference_script}")
        if not self.config_path.exists():
            raise RuntimeError(f"LTX pipeline config not found: {self.config_path}")

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

    def generate(self, data: Dict[str, object]) -> Dict[str, object]:
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
            str(data.get("output_dir") or os.getenv("KID_STUDIO_OUTPUT_PATH", "/workspace/outputs"))
        )
        output_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="kid-studio-ltx-") as temp_name:
            temp_dir = Path(temp_name)
            conditioning = self._resolve_conditioning_media(
                data.get("conditioning_media_paths") or [],
                data.get("conditioning_media_urls") or [],
                temp_dir,
            )

            start_frames = [int(value) for value in (data.get("conditioning_start_frames") or [])]
            strengths = [float(value) for value in (data.get("conditioning_strengths") or [])]

            if conditioning and not start_frames:
                start_frames = [0] * len(conditioning)
            if conditioning and len(start_frames) != len(conditioning):
                raise ValueError(
                    "conditioning_start_frames must match the number of conditioning media items"
                )
            if strengths and len(strengths) != len(conditioning):
                raise ValueError(
                    "conditioning_strengths must match the number of conditioning media items"
                )

            before = {path.resolve() for path in output_root.rglob("*.mp4")}

            command = [
                self.python,
                str(self.inference_script),
                "--prompt",
                prompt,
                "--pipeline_config",
                str(self.config_path),
                "--output_path",
                str(output_root),
                "--width",
                str(width),
                "--height",
                str(height),
                "--num_frames",
                str(num_frames),
                "--frame_rate",
                str(frame_rate),
                "--seed",
                str(seed),
                "--negative_prompt",
                negative_prompt,
            ]

            if bool(data.get("offload_to_cpu", False)):
                command.extend(["--offload_to_cpu", "true"])

            if conditioning:
                command.append("--conditioning_media_paths")
                command.extend(conditioning)
                command.append("--conditioning_start_frames")
                command.extend(str(value) for value in start_frames)
                if strengths:
                    command.append("--conditioning_strengths")
                    command.extend(str(value) for value in strengths)

            completed = subprocess.run(
                command,
                cwd=str(self.ltx_root),
                capture_output=True,
                text=True,
                timeout=int(data.get("timeout_seconds") or 7200),
                check=False,
            )

            if completed.returncode != 0:
                raise RuntimeError(
                    "LTX generation failed: "
                    + (completed.stderr[-6000:] or completed.stdout[-6000:] or "unknown error")
                )

            after = [path.resolve() for path in output_root.rglob("*.mp4")]
            generated = [path for path in after if path not in before]
            if not generated:
                raise RuntimeError(
                    "LTX completed without exposing a new MP4 in the configured output directory"
                )

            generated.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            output_path = generated[0]

            return {
                "ok": True,
                "engine": "ltx",
                "model": "LTX-Video 13B 0.9.8 distilled BF16",
                "output_path": str(output_path),
                "size_bytes": output_path.stat().st_size,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "seed": seed,
                "conditioning_count": len(conditioning),
                "stdout_tail": completed.stdout[-2000:],
            }

    def close(self) -> None:
        return None
