"""本地模型文件仓库访问层。"""

from pathlib import Path

from infer_nexus.catalog.models import ModelConfig
from infer_nexus.core.config import ModelStoreSettings
from infer_nexus.core.errors import ModelArtifactMissingError


class LocalModelStore:
    """负责模型路径解析、存在性校验与仓库目录映射。"""

    def __init__(self, root_dir: str | Path) -> None:
        """初始化本地模型根目录。"""
        self.root_dir = Path(root_dir).expanduser().resolve()

    @classmethod
    def from_settings(cls, settings: ModelStoreSettings) -> "LocalModelStore":
        """基于配置对象构造模型仓库实例。"""
        return cls(settings.root_dir)

    def resolve_model_path(self, model_path: str) -> Path:
        """将模型路径解析为绝对路径。"""
        candidate = Path(model_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.root_dir / candidate).resolve()

    def require_model_path(self, model: ModelConfig) -> Path:
        """要求模型路径存在，不存在则抛出业务异常。"""
        resolved_path = self.resolve_model_path(model.model_path)
        if not resolved_path.exists():
            raise ModelArtifactMissingError(model.name, model.model_path, str(resolved_path))
        return resolved_path

    def build_repo_subdir(self, repo_id: str) -> Path:
        """将 Hugging Face repo id 映射为本地分层目录。"""
        return Path(*repo_id.split("/"))

    def build_repo_target_dir(self, repo_id: str) -> Path:
        """构建指定 repo 对应的本地目标目录绝对路径。"""
        return (self.root_dir / self.build_repo_subdir(repo_id)).resolve()
