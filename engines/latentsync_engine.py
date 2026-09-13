class LatentSyncEngine:
    """Placeholder for the LatentSync inference engine.

    Real model loading and lip-sync are intentionally deferred until the
    worker/storage lifecycle is verified on a GPU Pod.
    """

    def __init__(self) -> None:
        raise RuntimeError("LatentSync inference is not enabled in the foundation build")

    def close(self) -> None:
        return None
