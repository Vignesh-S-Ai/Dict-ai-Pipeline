"""Brightness detection module."""

import numpy as np
import cv2
from typing import Tuple
from src.utils.config import config
from src.utils.logger import logger


def detect_brightness(image: np.ndarray) -> Tuple[float, str]:
    """Detect brightness level of an image.

    Args:
        image: Input image as numpy array (grayscale or color).

    Returns:
        Tuple containing:
            - mean: Mean grayscale value (0-255)
            - status: 'dark', 'normal', or 'bright'
    """
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        mean_brightness = np.mean(gray)

        thresholds = config.quality_thresholds
        if mean_brightness < thresholds.brightness_min:
            status = "dark"
        elif mean_brightness > thresholds.brightness_max:
            status = "bright"
        else:
            status = "normal"

        logger.info(
            f"Brightness detection: mean={mean_brightness:.2f}, status={status}"
        )

        return float(mean_brightness), status

    except Exception as e:
        logger.error(f"Error in brightness detection: {e}")
        return 0.0, "error"
