"""Logging utility for Document AI pipeline."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from src.utils.config import config


def setup_logger(name: str = __name__) -> logging.Logger:
    """Set up and return a configured logger.

    Args:
        name: Name of the logger.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.log_level.upper()))

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_path = Path(config.log_file)
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger(__name__)
