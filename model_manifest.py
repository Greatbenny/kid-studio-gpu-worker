from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ModelArtifact:
    repo_id: str
    filename: str
    sha256: str | None = None


@dataclass(frozen=True)
class ModelManifest:
    key: str
    version: str
    artifacts: Tuple[ModelArtifact, ...]


MODEL_MANIFESTS: Dict[str, ModelManifest] = {
    "ltx": ModelManifest(
        key="ltx",
        version="13b-0.9.8-distilled-bf16",
        artifacts=(
            ModelArtifact(
                repo_id="Lightricks/LTX-Video",
                filename="ltxv-13b-0.9.8-distilled.safetensors",
            ),
            ModelArtifact(
                repo_id="Lightricks/LTX-Video",
                filename="ltxv-spatial-upscaler-0.9.8.safetensors",
            ),
        ),
    ),
    "latentsync": ModelManifest(
        key="latentsync",
        version="1.6",
        artifacts=(
            ModelArtifact(
                repo_id="ByteDance/LatentSync-1.6",
                filename="latentsync_unet.pt",
            ),
            ModelArtifact(
                repo_id="ByteDance/LatentSync-1.6",
                filename="whisper/tiny.pt",
            ),
        ),
    ),
}
