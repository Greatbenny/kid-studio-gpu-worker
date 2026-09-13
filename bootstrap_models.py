import argparse
import hashlib
from pathlib import Path
from typing import Dict

from huggingface_hub import hf_hub_download

from model_manifest import MODEL_MANIFESTS
from storage import configure_cache_environment


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(model_key: str, verify: bool = True) -> Dict[str, object]:
    key = model_key.strip().lower()
    if key not in MODEL_MANIFESTS:
        raise ValueError(f"Unknown model: {key}")

    env = configure_cache_environment()
    models_root = Path(env["models_root"])
    manifest = MODEL_MANIFESTS[key]
    target_dir = models_root / key / manifest.version
    target_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for artifact in manifest.artifacts:
        local_path = target_dir / artifact.filename
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not local_path.exists():
            downloaded = hf_hub_download(
                repo_id=artifact.repo_id,
                filename=artifact.filename,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
            local_path = Path(downloaded)

        verified = None
        if verify and artifact.sha256:
            actual = sha256_file(local_path)
            if actual.lower() != artifact.sha256.lower():
                raise RuntimeError(
                    f"Checksum mismatch for {artifact.filename}: "
                    f"expected {artifact.sha256}, got {actual}"
                )
            verified = True

        results.append(
            {
                "repo_id": artifact.repo_id,
                "filename": artifact.filename,
                "path": str(local_path),
                "exists": local_path.exists(),
                "verified": verified,
            }
        )

    return {
        "ok": True,
        "model": key,
        "version": manifest.version,
        "persistent_storage": env["persistent"] == "true",
        "target_dir": str(target_dir),
        "artifacts": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore Kid Studio model files into the selected cache volume."
    )
    parser.add_argument("model", choices=sorted(MODEL_MANIFESTS.keys()))
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SHA256 verification where a checksum is pinned.",
    )
    args = parser.parse_args()

    result = ensure_model(args.model, verify=not args.no_verify)
    print(result)


if __name__ == "__main__":
    main()
