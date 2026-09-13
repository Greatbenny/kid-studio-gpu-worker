class LTXEngine:
    """Placeholder for the LTX inference engine.

    Real model loading and generation are intentionally deferred until the
    worker/storage lifecycle is verified on a GPU Pod.
    """

    def __init__(self) -> None:
        raise RuntimeError("LTX inference is not enabled in the foundation build")

    def close(self) -> None:
        return None
