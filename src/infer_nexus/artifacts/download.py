"""模型文件下载与模型注册片段生成工具。"""

from pathlib import Path
from textwrap import dedent

from infer_nexus.model_store import LocalModelStore


def download_huggingface_model(
    *,
    repo_id: str,
    target_dir: Path,
    endpoint: str,
    revision: str | None = None,
) -> Path:
    """下载 Hugging Face 模型到指定目录并返回落地路径。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install the 'artifacts' extra to use the download script."
        ) from exc

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target_dir,
        endpoint=endpoint,
    )
    return Path(downloaded_path).resolve()


def build_model_registration_snippet(
    *,
    name: str,
    alias: str,
    task: str,
    backend: str,
    model_path: str,
    tensor_parallel_size: int,
    cpu_per_replica: int | float,
    gpu_per_replica: int | float,
    min_replicas: int,
    max_replicas: int,
    dtype: str | None = None,
) -> str:
    """生成可粘贴到 `models.yaml` 的模型注册配置片段。"""
    dtype_line = f"\n    dtype: {dtype}" if dtype else ""
    return dedent(
        f"""
          - name: {name}
            alias: {alias}
            task: {task}
            backend: {backend}
            model_path: {model_path}{dtype_line}
            tensor_parallel_size: {tensor_parallel_size}
            cpu_per_replica: {cpu_per_replica}
            gpu_per_replica: {gpu_per_replica}
            min_replicas: {min_replicas}
            max_replicas: {max_replicas}
        """.rstrip()
    )


def prepare_huggingface_download(
    *,
    model_store: LocalModelStore,
    repo_id: str,
    revision: str | None = None,
    endpoint: str,
) -> Path:
    """根据本地模型仓库约定路径执行 Hugging Face 下载。"""
    target_dir = model_store.build_repo_target_dir(repo_id)
    return download_huggingface_model(
        repo_id=repo_id,
        target_dir=target_dir,
        endpoint=endpoint,
        revision=revision,
    )
