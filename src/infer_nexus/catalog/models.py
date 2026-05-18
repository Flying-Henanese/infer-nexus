"""Typed schemas for declarative model catalog entries."""

from pydantic import BaseModel, Field, model_validator

from infer_nexus.core.enums import BackendType, ModelStatus, TaskType


class ModelConfig(BaseModel):
    """Declarative model entry loaded from config/models.yaml."""

    name: str
    alias: str | None = None
    task: TaskType
    backend: BackendType = BackendType.VLLM
    model_path: str
    dtype: str | None = None
    tensor_parallel_size: int = Field(ge=1)
    max_model_len: int | None = None
    cpu_per_replica: int | float = Field(gt=0)
    gpu_per_replica: int | float = Field(ge=0)
    min_replicas: int = Field(ge=0)
    max_replicas: int = Field(ge=1)
    capabilities: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    status: ModelStatus = ModelStatus.UNKNOWN

    @model_validator(mode="after")
    def validate_replica_bounds(self) -> "ModelConfig":
        """Ensure replica bounds are internally consistent."""
        if self.max_replicas < self.min_replicas:
            raise ValueError("max_replicas must be >= min_replicas")
        return self


class ModelCatalogFile(BaseModel):
    """Top-level catalog file schema."""

    models: list[ModelConfig]
