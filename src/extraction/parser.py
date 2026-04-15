import re
from typing import Dict, List
from src.utils.logger import logger


def clean_text(text: str) -> str:
    """Clean OCR text by removing noise characters."""
    text = text.replace("\n", " ")
    text = re.sub(r"[^\w\s:/-]", "", text)  # remove weird symbols
    text = re.sub(r"\s+", " ", text)  # normalize spaces
    return text.strip()


def extract_names(text: str) -> List[str]:
    """Extract probable human names."""
    candidates = re.findall(r"\b[A-Z][a-z]{2,}\s[A-Z][a-z]{2,}\b", text)

    # 🚫 Filter out common non-name words
    stopwords = {
        "Our", "Solution", "Saas", "April", "March",
        "Showcasing", "Company", "Project"
    }

    valid_names = []
    for name in candidates:
        words = name.split()
        if not any(word in stopwords for word in words):
            valid_names.append(name)

    return list(set(valid_names))


def extract_dates(text: str) -> List[str]:
    """Extract dates in multiple formats."""
    
    patterns = [
        r"\b\d{2}/\d{2}/\d{4}\b",        # 12/05/2023
        r"\b\d{4}-\d{2}-\d{2}\b",        # 2023-05-12
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4}\b"  # Apr 2025
    ]

    dates = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text))

    return list(set(dates))


def extract_ids(text: str) -> List[str]:
    """Extract alphanumeric IDs."""
    return re.findall(r"\b[A-Z0-9]{6,}\b", text)


def extract_emails(text: str) -> List[str]:
    """Extract email addresses."""
    pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    return list(set(re.findall(pattern, text)))


def extract_phones(text: str) -> List[str]:
    """Extract phone numbers (basic international/national)."""
    # Matches formats like +1-234-567-8900, (123) 456-7890, 123-456-7890
    pattern = r"\+?\d{1,3}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
    
    candidates = re.findall(pattern, text)
    # Filter out obvious non-phones (must have enough digits)
    phones = [p.strip() for p in candidates if len(re.sub(r"\D", "", p)) >= 7]
    return list(set(phones))


def extract_amounts(text: str) -> List[str]:
    """Extract currency amounts with symbols."""
    # Matches $1,000.00, €50, Rs. 100
    pattern = r"(?:[\$€£]|Rs\.?)\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?"
    return list(set(re.findall(pattern, text)))


def parse_extracted_text(text: str) -> Dict[str, List[str]]:
    """Main parsing function."""

    logger.info("Starting structured data extraction")

    cleaned = clean_text(text)

    names = extract_names(cleaned)
    dates = extract_dates(cleaned)
    ids = extract_ids(cleaned)
    emails = extract_emails(cleaned)
    phones = extract_phones(cleaned)
    amounts = extract_amounts(text)  # Run on raw text to keep symbols if any

    logger.info(
        f"Parsing results: {len(names)} names, {len(dates)} dates, {len(ids)} IDs, {len(emails)} emails, {len(phones)} phones, {len(amounts)} amounts"
    )

    return {
        "name": names,
        "dates": dates,
        "ids": ids,
        "emails": emails,
        "phones": phones,
        "amounts": amounts,
    }


def detect_language(text: str) -> str:
    """Basic language detection."""
    if re.search(r"[a-zA-Z]", text):
        return "english"
    return "unknown"