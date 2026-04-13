"""Configuration settings for Document AI pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QualityThresholds:
    """Quality detection thresholds."""

    blur_threshold: float = 60.0
    brightness_min: float = 80.0
    brightness_max: float = 200.0
    noise_threshold: float = 80.0


@dataclass
class Config:
    """Main configuration for the pipeline."""

    input_dir: Path = Path("data/input")
    output_dir: Path = Path("data/output")
    output_file: str = "result.json"
    log_file: str = "document_ai.log"
    log_level: str = "INFO"

    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)


config = Config()
