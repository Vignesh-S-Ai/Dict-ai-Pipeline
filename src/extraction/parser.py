"""Text parsing module for extracting structured data."""

import re
from typing import Dict, List
from src.utils.logger import logger


def parse_extracted_text(text: str) -> Dict[str, List[str]]:
    """Parse extracted text to find names, dates, and IDs.

    Args:
        text: Raw OCR extracted text.

    Returns:
        Dictionary with lists of found names, dates, and IDs.
    """
    result = {"name": [], "dates": [], "ids": []}

    name_pattern = r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b"
    names = re.findall(name_pattern, text)
    result["name"] = [f"{n[0]} {n[1]}" for n in names]

    date_pattern = r"\b(\d{2}/\d{2}/\d{4})\b"
    dates = re.findall(date_pattern, text)
    result["dates"] = dates

    id_pattern = r"\b([A-Z]{1,3}\d{6,10})\b"
    ids = re.findall(id_pattern, text)
    result["ids"] = ids

    logger.info(
        f"Parsing results: {len(result['name'])} names, "
        f"{len(result['dates'])} dates, {len(result['ids'])} IDs"
    )

    return result


def detect_language(text: str) -> str:
    """Detect if text contains Telugu characters.

    Args:
        text: Input text to analyze.

    Returns:
        Language code: 'telugu', 'english', or 'mixed'.
    """
    telugu_range = range(0x0C00, 0x0C7F)
    telugu_chars = sum(1 for char in text if ord(char) in telugu_range)

    if telugu_chars > 10:
        return "telugu"
    elif telugu_chars > 0:
        return "mixed"
    else:
        return "english"
