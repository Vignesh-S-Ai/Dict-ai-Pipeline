"""Image preprocessing utilities for OCR."""

import numpy as np
import cv2
from typing import Optional
from src.utils.logger import logger


def preprocess_image(image: np.ndarray, apply_denoising: bool = False) -> np.ndarray:
    """Preprocess image for OCR with grayscale conversion and binarization.

    Args:
        image: Input image as numpy array.
        apply_denoising: Whether to apply denoising (default: False).

    Returns:
        Preprocessed binary image ready for OCR.
    """
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        logger.info("Image converted to grayscale")

        if apply_denoising:
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            logger.info("Denoising applied")

        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        logger.info("Otsu's binarization applied")

        return binary

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise
