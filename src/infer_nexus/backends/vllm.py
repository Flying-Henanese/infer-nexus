from typing import Any

from infer_nexus.backends.base import InferenceBackend
from infer_nexus.catalog.models import ModelConfig


class VLLMBackend(InferenceBackend):
    def build_runtime_spec(self, model: ModelConfig) -> dict[str, Any]:
        return {
            "backend": "vllm",
            "model_path": model.model_path,
            "tensor_parallel_size": model.tensor_parallel_size,
            "dtype": model.dtype,
            "gpu_per_replica": model.gpu_per_replica,
            "cpu_per_replica": model.cpu_per_replica,
        }
