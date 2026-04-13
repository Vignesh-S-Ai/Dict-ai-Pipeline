"""Noise detection module."""

import numpy as np
import cv2
from typing import Tuple
from src.utils.config import config
from src.utils.logger import logger


def detect_noise(image: np.ndarray) -> Tuple[float, str]:
    """Detect noise level in an image using standard deviation.

    Args:
        image: Input image as numpy array (grayscale or color).

    Returns:
        Tuple containing:
            - std_dev: Standard deviation of pixel values
            - status: 'low_noise', 'medium_noise', or 'high_noise'
    """
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        std_dev = np.std(gray)

        threshold = config.quality_thresholds.noise_threshold
        if std_dev < threshold * 0.5:
            status = "low_noise"
        elif std_dev < threshold:
            status = "medium_noise"
        else:
            status = "high_noise"

        logger.info(f"Noise detection: std_dev={std_dev:.2f}, status={status}")

        return float(std_dev), status

    except Exception as e:
        logger.error(f"Error in noise detection: {e}")
        return 0.0, "error"
