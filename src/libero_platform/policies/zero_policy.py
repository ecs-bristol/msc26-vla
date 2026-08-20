from __future__ import annotations

import numpy as np

from libero_platform.policies.base import PolicyAdapter, PolicyRequest, PolicyResponse


class ZeroPolicyAdapter(PolicyAdapter):
    def __init__(self, model_key: str) -> None:
        self._model_key = model_key

    def predict(self, request: PolicyRequest) -> PolicyResponse:
        del request
        return PolicyResponse(
            action=np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
            inference_ms=0.0,
            model_key=self._model_key,
            device="cpu",
        )
