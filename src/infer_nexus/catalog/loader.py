from pathlib import Path

import yaml

from infer_nexus.catalog.models import ModelCatalogFile
from infer_nexus.core.errors import ConfigError


def load_model_catalog(path: str | Path) -> ModelCatalogFile:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"model catalog file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return ModelCatalogFile.model_validate(raw)
