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
        version="13b-0.9.8-distilled-fp8",
        artifacts=(
            ModelArtifact(
                repo_id="Lightricks/LTX-Video",
                filename="ltxv-13b-0.9.8-distilled-fp8.safetensors",
                sha256="111a3d07baa17f520e98b571e7916139ae0865c9a24b7534529d6b9e74264db3",
            ),
            ModelArtifact(
                repo_id="Lightricks/LTX-Video",
                filename="ltxv-spatial-upscaler-0.9.8.safetensors",
                sha256="5b076031c6f860db9037a54f3bb819f10bfb5532ea26a6d30062292428a0c208",
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
