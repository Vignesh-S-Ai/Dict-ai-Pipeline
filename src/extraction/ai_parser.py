"""Advanced OCR extraction using Google's Gemini Pro Vision."""

import os
import json
from typing import Dict, Any, Optional
import google.generativeai as genai
from PIL import Image
import numpy as np
from src.utils.logger import logger
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    # Use gemini-1.5-flash for speed and vision capabilities
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    logger.warning("GEMINI_API_KEY not found in environment. AI parsing will be disabled.")
    model = None

PROMPT = """
You are a document extraction expert. 
Analyze the provided document image and extract the following information in a valid JSON format:
- name: Full name of the individual
- dates: Any dates found in the document (list)
- ids: Any identification numbers (passport, DL, etc.)
- addresses: Any addresses found
- document_type: Identify the type of document (e.g., Passport, ID Card, Invoice, etc.)
- raw_text: A summary of the main text content

Return ONLY the JSON object.
"""

def extract_with_gemini(image: np.ndarray) -> Optional[Dict[str, Any]]:
    """Extract structured data using Gemini Vision."""
    if model is None:
        logger.error("Gemini model not initialized. Missing API Key.")
        return None

    try:
        # Convert OpenCV BGR to RGB PIL Image
        import cv2
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)

        logger.info("Sending image to Gemini for strict JSON analysis...")
        
        # Next-level constraint: Force JSON mime-type so we never get markdown
        config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,  # Low temperature for precise data extraction
        )
        
        response = model.generate_content(
            [PROMPT, pil_img],
            generation_config=config
        )
        
        parsed_data = json.loads(response.text)
        logger.info("Successfully extracted structured data with Gemini")
        return parsed_data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini JSON output: {e}\nRaw output: {response.text}")
        return None
    except Exception as e:
        logger.error(f"Gemini extraction failed: {e}")
        return None

def detect_language_ai(text: str) -> str:
    """Use Gemini to detect language if needed (fallback to simple logic)."""
    # Placeholder for more complex logic
    return "English" 
