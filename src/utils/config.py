import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass
class QualityThresholds:
    """Quality detection thresholds."""

    blur_threshold: float = float(os.getenv("BLUR_THRESHOLD", 60.0))
    brightness_min: float = float(os.getenv("BRIGHTNESS_MIN", 80.0))
    brightness_max: float = float(os.getenv("BRIGHTNESS_MAX", 200.0))
    noise_threshold: float = float(os.getenv("NOISE_THRESHOLD", 80.0))


@dataclass
class Config:
    """Main configuration for the pipeline."""

    input_dir: Path = Path(os.getenv("INPUT_DIR", "data/input"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "data/output"))
    output_file: str = os.getenv("OUTPUT_FILE", "result.json")
    log_file: str = os.getenv("LOG_FILE", "document_ai.log")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)

    # API Keys
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")


config = Config()
