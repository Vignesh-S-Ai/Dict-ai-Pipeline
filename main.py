"""Main entry point for Document AI pipeline."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np

from src.utils.config import config
from src.utils.logger import logger
from src.utils.preprocess import preprocess_image

from src.quality.blur import detect_blur
from src.quality.brightness import detect_brightness
from src.quality.noise import detect_noise

from src.extraction.ocr import extract_text, get_confidence, extract_text_with_confidence
from src.extraction.parser import parse_extracted_text, detect_language


def load_image(image_path: str) -> np.ndarray:
    """Load an image from file path.

    Args:
        image_path: Path to the image file.

    Returns:
        Image as numpy array.

    Raises:
        FileNotFoundError: If image cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    logger.info(f"Loaded image: {image_path}")
    return img


def analyze_quality(image: np.ndarray) -> Dict[str, Any]:
    """Run all quality analysis modules on the image.

    Args:
        image: Input image.

    Returns:
        Dictionary with quality analysis results.
    """
    logger.info("Starting quality analysis")

    blur_score, blur_status = detect_blur(image)
    brightness_score, brightness_status = detect_brightness(image)
    noise_score, noise_status = detect_noise(image)

    return {
        "blur": {"score": blur_score, "status": blur_status},
        "brightness": {"score": brightness_score, "status": brightness_status},
        "noise": {"score": noise_score, "status": noise_status},
    }


def should_reject(quality: Dict[str, Any]) -> bool:
    """Decide whether to reject image based on quality."""
    if quality["blur"]["status"] == "blurry":
        return True
    if quality["noise"]["status"] == "high_noise":
        return True
    return False


def run_pipeline(image_path: str, apply_denoising: bool = False) -> Dict[str, Any]:
    """Run the complete Document AI pipeline with smart decision logic."""

    logger.info(f"Starting pipeline for: {image_path}")

    try:
        image = load_image(image_path)
        logger.info("Image loaded successfully")
    except Exception as e:
        logger.error(f"Image load failed: {e}")
        return {"error": str(e)}

    # 🔍 Step 1: Quality Check
    quality_results = analyze_quality(image)

    # 🚫 Step 2: Auto Reject (NEW 🔥)
    if should_reject(quality_results):
        logger.warning("Image rejected due to poor quality")

        return {
            "status": "Rejected - Poor Quality",
            "reason": "Image is blurry or too noisy",
            "quality": quality_results,
        }

    # 🧼 Step 3: Preprocessing
    try:
        preprocessed = preprocess_image(image, apply_denoising=apply_denoising)
        logger.info("Image preprocessed successfully")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return {"error": "Preprocessing failed", "quality": quality_results}

    # 🧠 Step 4: OCR + Confidence (UPGRADED)
    extracted_text, confidence = extract_text_with_confidence(preprocessed)

    if not extracted_text.strip():
        logger.warning("OCR extracted no text")
        return {
            "status": "OCR Failed",
            "quality": quality_results
        }

    # 🚫 Step 5: Confidence Guard (NEW 🔥)
    if confidence is not None and confidence < 60:
        logger.warning(f"Low OCR confidence: {confidence}")

    # Step 6: Parsing & Metadata
    parsed_data = parse_extracted_text(extracted_text)
    language = detect_language(extracted_text)

    return {
        "image_path": image_path,
        "quality": quality_results,
        "extracted_text": extracted_text.strip(),
        "parsed_data": parsed_data,
        "language": language,
        "confidence": confidence,
        "status": "Success"
    }


def save_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save results to JSON file.

    Args:
        results: Pipeline results.
        output_path: Path to output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_path}")


def print_results(results: Dict[str, Any]) -> None:
    """Print results to console.

    Args:
        results: Pipeline results.
    """
    print("\n" + "=" * 50)
    print("DOCUMENT AI PIPELINE RESULTS")
    print("=" * 50)
    print(json.dumps(results, indent=2))
    print("=" * 50 + "\n")


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Document Image Quality Analysis & OCR Extraction Pipeline"
    )
    parser.add_argument("--image", required=True, help="Path to input image file")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output JSON file (default: data/output/result.json)",
    )
    parser.add_argument(
        "--denoise", action="store_true", help="Apply denoising during preprocessing"
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        print(f"Error: Image file not found: {image_path}")
        return 1

    results = run_pipeline(str(image_path), apply_denoising=args.denoise)

    output_path = (
        Path(args.output) if args.output else config.output_dir / config.output_file
    )
    save_results(results, output_path)
    print_results(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())