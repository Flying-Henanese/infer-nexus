from pathlib import Path

from infer_nexus.catalog.models import ModelConfig
from infer_nexus.core.config import ModelStoreSettings
from infer_nexus.core.errors import ModelArtifactMissingError


class LocalModelStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()

    @classmethod
    def from_settings(cls, settings: ModelStoreSettings) -> "LocalModelStore":
        return cls(settings.root_dir)

    def resolve_model_path(self, model_path: str) -> Path:
        candidate = Path(model_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.root_dir / candidate).resolve()

    def require_model_path(self, model: ModelConfig) -> Path:
        resolved_path = self.resolve_model_path(model.model_path)
        if not resolved_path.exists():
            raise ModelArtifactMissingError(model.name, model.model_path, str(resolved_path))
        return resolved_path

    def build_repo_subdir(self, repo_id: str) -> Path:
        return Path(*repo_id.split("/"))

    def build_repo_target_dir(self, repo_id: str) -> Path:
        return (self.root_dir / self.build_repo_subdir(repo_id)).resolve()
