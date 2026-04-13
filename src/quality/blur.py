"""Blur detection module using Laplacian variance."""

import numpy as np
import cv2
from typing import Tuple
from src.utils.config import config
from src.utils.logger import logger


def detect_blur(image: np.ndarray) -> Tuple[float, str]:
    """Detect blur in an image using Laplacian variance.

    Args:
        image: Input image as numpy array (grayscale or color).

    Returns:
        Tuple containing:
            - variance: Laplacian variance score (higher = sharper)
            - status: 'sharp' or 'blurry'
    """
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        threshold = config.quality_thresholds.blur_threshold
        status = "sharp" if laplacian_var >= threshold else "blurry"

        logger.info(f"Blur detection: variance={laplacian_var:.2f}, status={status}")

        return float(laplacian_var), status

    except Exception as e:
        logger.error(f"Error in blur detection: {e}")
        return 0.0, "error"
