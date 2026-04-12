"""OCR extraction module using pytesseract."""

import pytesseract
import numpy as np
from typing import Optional, Tuple
from src.utils.logger import logger

# 🔥 IMPORTANT: Explicit path fix (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text(image: np.ndarray, lang: str = "eng") -> str:
    """Extract text from an image using pytesseract OCR.

    Args:
        image: Preprocessed binary image.
        lang: Language code for OCR (default: 'eng').

    Returns:
        Extracted text string.
    """
    try:
        logger.info(f"Starting OCR extraction with lang={lang}")

        # Optional config for better OCR accuracy
        custom_config = r'--oem 3 --psm 6'

        text = pytesseract.image_to_string(
            image,
            lang=lang,
            config=custom_config
        )

        text = text.strip()

        if text:
            logger.info(f"OCR extraction successful, extracted {len(text)} characters")
        else:
            logger.warning("OCR extracted empty text")

        return text

    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract is not installed or not found in PATH")
        return ""

    except pytesseract.TesseractError as e:
        logger.error(f"Tesseract OCR error: {e}")
        return ""

    except Exception as e:
        logger.error(f"Unexpected OCR error: {e}")
        return ""


def get_confidence(image: np.ndarray) -> Optional[float]:
    """Get OCR confidence score for the image.

    Args:
        image: Preprocessed image.

    Returns:
        Average confidence score (0-100) or None if unavailable.
    """
    try:
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT
        )

        confidences = []

        for conf in data["conf"]:
            try:
                conf_val = float(conf)
                if conf_val >= 0:
                    confidences.append(conf_val)
            except ValueError:
                continue

        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            logger.info(f"OCR average confidence: {avg_confidence:.2f}")
            return round(avg_confidence, 2)

        logger.warning("No valid confidence values found")
        return None

    except Exception as e:
        logger.error(f"Error getting confidence: {e}")
        return None


def extract_text_with_confidence(image: np.ndarray, lang: str = "eng") -> Tuple[str, Optional[float]]:
    """Extract text along with confidence score.

    Args:
        image: Preprocessed image.
        lang: OCR language.

    Returns:
        Tuple of (text, confidence score)
    """
    text = extract_text(image, lang)
    confidence = get_confidence(image)

    return text, confidence