"""Core data contracts used across the system.

These contracts are intentionally JSON-friendly and must not contain tensors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol, Sequence


class Batch(Protocol):
    """Protocol for batches passed through training and forensics.

    Each batch must include stable sample IDs for blacklist actions.
    """

    sample_id: Sequence[str]


@dataclass
class MetricsWindow:
    loss_history: list[float] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)


@dataclass
class Anomaly:
    type: str | None = None  # "loss_spike" | "grad_explosion" | "nan" | None
    step: int | None = None


@dataclass
class OutlierStats:
    median_grad_norm: float = 0.0
    max_grad_norm: float = 0.0


@dataclass
class DiagnosisReport:
    batch_id: str
    outlier_indices: list[int] = field(default_factory=list)
    outlier_stats: OutlierStats = field(default_factory=OutlierStats)
    suspected_samples: list[str] = field(default_factory=list)


@dataclass
class CurriculumMix:
    domain_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class HumanInterrupt:
    pending: bool = False
    reason: str | None = None


@dataclass
class State:
    run_id: str
    step: int = 0
    last_checkpoint_path: str = ""
    metrics_window: MetricsWindow = field(default_factory=MetricsWindow)
    anomaly: Anomaly = field(default_factory=Anomaly)
    diagnosis_report: DiagnosisReport | None = None
    repair_attempts: int = 0
    blacklist: list[str] = field(default_factory=list)
    curriculum_mix: CurriculumMix = field(default_factory=CurriculumMix)
    human_interrupt: HumanInterrupt = field(default_factory=HumanInterrupt)

    def to_dict(self) -> dict:
        """Serialize state to a JSON-friendly dict."""

        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "State":
        """Deserialize state from a dict produced by to_dict()."""

        metrics_window = MetricsWindow(**data.get("metrics_window", {}))
        anomaly = Anomaly(**data.get("anomaly", {}))
        diagnosis_report_raw = data.get("diagnosis_report")
        diagnosis_report = None
        if diagnosis_report_raw is not None:
            outlier_stats = OutlierStats(**diagnosis_report_raw.get("outlier_stats", {}))
            diagnosis_report = DiagnosisReport(
                batch_id=diagnosis_report_raw.get("batch_id", ""),
                outlier_indices=list(diagnosis_report_raw.get("outlier_indices", [])),
                outlier_stats=outlier_stats,
                suspected_samples=list(diagnosis_report_raw.get("suspected_samples", [])),
            )

        curriculum_mix = CurriculumMix(**data.get("curriculum_mix", {}))
        human_interrupt = HumanInterrupt(**data.get("human_interrupt", {}))

        return State(
            run_id=data.get("run_id", ""),
            step=int(data.get("step", 0)),
            last_checkpoint_path=data.get("last_checkpoint_path", ""),
            metrics_window=metrics_window,
            anomaly=anomaly,
            diagnosis_report=diagnosis_report,
            repair_attempts=int(data.get("repair_attempts", 0)),
            blacklist=list(data.get("blacklist", [])),
            curriculum_mix=curriculum_mix,
            human_interrupt=human_interrupt,
        )
