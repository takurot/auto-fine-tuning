"""Configuration schema for auto-fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    steps_per_window: int = Field(100, ge=1)
    batch_size: int = Field(1024, ge=1)
    lr: float = Field(3e-5, gt=0)
    grad_clip: float = Field(1.0, gt=0)


class MonitoringConfig(BaseModel):
    spike_sigma: float = Field(3.0, gt=0)
    grad_norm_threshold: float = Field(1.0e4, gt=0)
    window_size: int = Field(1000, ge=10)


class ForensicsConfig(BaseModel):
    outlier_ratio: float = Field(100.0, gt=1.0)
    zscore_threshold: float = Field(6.0, gt=0)


class RecoveryConfig(BaseModel):
    max_retries: int = Field(3, ge=0)


class CheckpointConfig(BaseModel):
    interval: int = Field(100, ge=1)
    keep_last_n: int = Field(5, ge=1)
    async_save: bool = True
    dir: Path = Path("checkpoints")


class CurriculumConfig(BaseModel):
    eval_interval: int = Field(1000, ge=1)
    domains: list[str] = Field(default_factory=lambda: ["general"])
    weights: dict[str, float] = Field(default_factory=lambda: {"general": 1.0})


class NotificationsConfig(BaseModel):
    webhook_url: str | None = None
    slack_channel: str | None = None


class PersistenceConfig(BaseModel):
    backend: Literal["sqlite", "postgres"] = "sqlite"
    database_url: str = "sqlite:///aft.db"


class MetricsConfig(BaseModel):
    stdout: bool = True
    jsonl_path: Path = Path("runs/metrics.jsonl")


class AppConfig(BaseModel):
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    forensics: ForensicsConfig = Field(default_factory=ForensicsConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load config from a YAML file into AppConfig.

    Note: YAML parsing is deferred to avoid hard dependency until needed.
    """

    path = Path(path)
    data = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to load config files. Install pyyaml."
        ) from exc

    parsed = yaml.safe_load(data) or {}
    return AppConfig.model_validate(parsed)
