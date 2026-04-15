"""FastAPI application for Document AI Dashboard."""

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
import shutil
import json

from main import run_pipeline
from src.utils.config import config
from src.utils.logger import logger
from src.extraction.ai_parser import extract_with_gemini

app = FastAPI(title="Vision-Pro Document AI")

# Create directories if they don't exist
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files (we'll generate images/css here)
app.mount("/data", StaticFiles(directory="data"), name="data")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Return the premium dashboard UI."""
    return DASHBOARD_HTML

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    denoise: bool = Form(False),
    use_ai: bool = Form(False)
):
    """Process uploaded document and return results."""
    try:
        # Save file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run pipeline
        results = run_pipeline(str(file_path), apply_denoising=denoise)
        
        # Optionally run Gemini AI parsing
        if use_ai and results.get("status") == "Success":
            img = cv2.imread(str(file_path))
            ai_data = extract_with_gemini(img)
            if ai_data:
                results["ai_parsed_data"] = ai_data
                results["status"] = "Success (AI Enhanced)"

        return results
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    """Get list of previously processed results."""
    output_dir = Path("data/output")
    if not output_dir.exists():
        return []
    
    # This is a simplification; in a real app, we'd use a DB
    results = []
    for f in output_dir.glob("*.json"):
        with open(f, "r") as jf:
            results.append(json.load(jf))
    return results

# Premium Glassmorphic Dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vision-Pro | Document AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.5);
            --bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #10b981;
            --error: #ef4444;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Outfit', sans-serif;
        }

        body {
            background: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.1) 0px, transparent 50%);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        header {
            padding: 2rem;
            text-align: center;
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(to right, #818cf8, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .container {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 1rem;
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 2rem;
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 30px 60px -12px rgba(99, 102, 241, 0.2);
        }

        .upload-area {
            border: 2px dashed rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 3rem 1rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .upload-area:hover {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.05);
        }

        .upload-area i {
            font-size: 3rem;
            margin-bottom: 1rem;
            display: block;
        }

        input[type="file"] {
            display: none;
        }

        .btn {
            background: var(--primary);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-top: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 0 15px var(--primary-glow);
        }

        .btn:hover {
            filter: brightness(1.1);
            transform: scale(1.02);
        }

        .options {
            margin-top: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
            color: var(--text-muted);
        }

        .checkbox-group input {
            accent-color: var(--primary);
            width: 1.1rem;
            height: 1.1rem;
        }

        #result-container {
            display: none;
            animation: fadeIn 0.5s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 99px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        .status-success { background: rgba(16, 185, 129, 0.2); color: var(--accent); }
        .status-error { background: rgba(239, 68, 68, 0.2); color: var(--error); }

        .quality-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 1.5rem 0;
        }

        .quality-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
        }

        .quality-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; }
        .quality-value { font-size: 1.1rem; font-weight: 600; margin-top: 0.25rem; }

        .extracted-text {
            background: rgba(0, 0, 0, 0.3);
            padding: 1rem;
            border-radius: 12px;
            font-family: monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .json-viewer {
            margin-top: 1rem;
            padding: 1.5rem;
            background: rgba(15, 23, 42, 0.8);
            border-radius: 12px;
            font-size: 0.9rem;
            overflow-x: auto;
            position: relative;
        }

        .json-key { color: #818cf8; font-weight: 600; }
        .json-string { color: #34d399; }
        .json-number { color: #fbbf24; }
        .json-boolean { color: #f472b6; }
        .json-null { color: #94a3b8; }

        .copy-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255, 255, 255, 0.1);
            border: none;
            color: #fff;
            padding: 5px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.3s ease;
        }
        
        .copy-btn:hover {
            background: var(--primary);
        }

        .ai-data {
            border-left: 4px solid var(--primary);
            margin-top: 1.5rem;
            padding-left: 1rem;
            animation: fadeInRight 0.5s ease;
        }
        
        @keyframes fadeInRight {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }

        .parsed-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-top: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .parsed-item {
            background: rgba(99, 102, 241, 0.05);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(99, 102, 241, 0.2);
        }
        
        .parsed-item strong {
            display: block;
            font-size: 0.8rem;
            color: #818cf8;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
        }

        #preview-img {
            max-width: 100%;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        .loader {
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s linear infinite;
            display: none;
            margin: 0 auto;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        footer {
            margin-top: auto;
            padding: 2rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
        }
    </style>
</head>
<body>
    <header>
        <h1>VISION-PRO</h1>
        <p style="color: var(--text-muted)">State-of-the-art Document Intelligence Pipeline</p>
    </header>

    <div class="container">
        <!-- Input Section -->
        <div class="card">
            <h2 style="margin-bottom: 1.5rem">Intelligence Control</h2>
            <div class="upload-area" id="drop-zone">
                <span style="font-size: 3rem">📄</span>
                <p style="margin-top: 1rem">Drop document or click to upload</p>
                <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem">PNG, JPG, WebP supported</p>
                <input type="file" id="file-input" accept="image/*">
            </div>

            <div class="options">
                <div class="checkbox-group">
                    <input type="checkbox" id="denoise-check">
                    <label for="denoise-check">Advanced De-noising (Non-Local)</label>
                </div>
                <div class="checkbox-group">
                    <input type="checkbox" id="ai-check" checked>
                    <label for="ai-check">Gemini AI Synthesis (Highly Recommended)</label>
                </div>
            </div>

            <button class="btn" id="process-btn">
                <span id="btn-text">RUN PIPELINE</span>
                <div class="loader" id="btn-loader"></div>
            </button>
        </div>

        <!-- Output Section -->
        <div class="card">
            <div id="welcome-message" style="text-align: center; padding: 4rem 0;">
                <span style="font-size: 4rem; opacity: 0.2">🔍</span>
                <p style="color: var(--text-muted); margin-top: 1rem">Awaiting document for analysis...</p>
            </div>

            <div id="result-container">
                <div class="status-badge status-success" id="status-text">Pipeline Success</div>
                
                <img id="preview-img" src="" alt="Preview">

                <div class="quality-grid">
                    <div class="quality-item">
                        <div class="quality-label">Blur</div>
                        <div class="quality-value" id="blur-val">-</div>
                    </div>
                    <div class="quality-item">
                        <div class="quality-label">Brightness</div>
                        <div class="quality-value" id="brightness-val">-</div>
                    </div>
                    <div class="quality-item">
                        <div class="quality-label">Noise</div>
                        <div class="quality-value" id="noise-val">-</div>
                    </div>
                </div>

                <div style="margin-top: 1.5rem" id="parsed-data-section">
                    <h3 style="margin-bottom: 0.5rem; font-size: 1rem">Fast Regex Extraction</h3>
                    <div class="parsed-grid">
                        <div class="parsed-item"><strong>Names</strong> <span id="parsed-names">-</span></div>
                        <div class="parsed-item"><strong>Dates</strong> <span id="parsed-dates">-</span></div>
                        <div class="parsed-item"><strong>Emails</strong> <span id="parsed-emails">-</span></div>
                        <div class="parsed-item"><strong>Phones</strong> <span id="parsed-phones">-</span></div>
                        <div class="parsed-item"><strong>Amounts</strong> <span id="parsed-amounts">-</span></div>
                        <div class="parsed-item"><strong>IDs</strong> <span id="parsed-ids">-</span></div>
                    </div>
                </div>

                <div style="margin-top: 1.5rem">
                    <h3 style="margin-bottom: 0.5rem; font-size: 1rem">Extracted Text (OCR)</h3>
                    <div class="extracted-text" id="raw-text-view"></div>
                </div>

                <div class="ai-data" id="ai-section" style="display: none;">
                    <h3 style="margin-bottom: 0.5rem; font-size: 1rem; color: #818cf8;">✨ AI Structured Insights</h3>
                    <div class="json-viewer">
                        <button class="copy-btn" onclick="copyJson()">Copy</button>
                        <pre id="ai-json-view"></pre>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer>
        &copy; 2026 Vision-Pro Document AI • Powered by Gemini Flash
    </footer>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const processBtn = document.getElementById('process-btn');
        const btnText = document.getElementById('btn-text');
        const btnLoader = document.getElementById('btn-loader');
        const resultContainer = document.getElementById('result-container');
        const welcomeMessage = document.getElementById('welcome-message');

        dropZone.onclick = () => fileInput.click();

        fileInput.onchange = (e) => {
            if (e.target.files.length > 0) {
                dropZone.querySelector('p').innerText = e.target.files[0].name;
                dropZone.style.borderColor = 'var(--primary)';
            }
        };

        processBtn.onclick = async () => {
            const file = fileInput.files[0];
            if (!file) {
                alert('Please select a file first.');
                return;
            }

            // UI State
            btnText.style.display = 'none';
            btnLoader.style.display = 'block';
            processBtn.disabled = true;

            const formData = new FormData();
            formData.append('file', file);
            formData.append('denoise', document.getElementById('denoise-check').checked);
            formData.append('use_ai', document.getElementById('ai-check').checked);

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                // Update UI with results
                welcomeMessage.style.display = 'none';
                resultContainer.style.display = 'block';

                // Preview Image
                document.getElementById('preview-img').src = URL.createObjectURL(file);
                
                // Status
                document.getElementById('status-text').innerText = data.status || 'Processed';
                document.getElementById('status-text').className = 'status-badge ' + 
                    (data.error ? 'status-error' : 'status-success');

                // Quality
                if (data.quality) {
                    document.getElementById('blur-val').innerText = data.quality.blur.status;
                    document.getElementById('brightness-val').innerText = data.quality.brightness.status;
                    document.getElementById('noise-val').innerText = data.quality.noise.status;
                }

                // Regex Parsed Data
                if (data.parsed_data) {
                    document.getElementById('parsed-names').innerText = (data.parsed_data.name || []).join(', ') || 'None';
                    document.getElementById('parsed-dates').innerText = (data.parsed_data.dates || []).join(', ') || 'None';
                    document.getElementById('parsed-emails').innerText = (data.parsed_data.emails || []).join(', ') || 'None';
                    document.getElementById('parsed-phones').innerText = (data.parsed_data.phones || []).join(', ') || 'None';
                    document.getElementById('parsed-amounts').innerText = (data.parsed_data.amounts || []).join(', ') || 'None';
                    document.getElementById('parsed-ids').innerText = (data.parsed_data.ids || []).join(', ') || 'None';
                }

                // OCR Text
                document.getElementById('raw-text-view').innerText = data.extracted_text || 'None';

                // AI Section
                if (data.ai_parsed_data) {
                    document.getElementById('ai-section').style.display = 'block';
                    document.getElementById('ai-json-view').innerHTML = syntaxHighlight(JSON.stringify(data.ai_parsed_data, null, 2));
                    // Store for copy button
                    window.currentAiJson = JSON.stringify(data.ai_parsed_data, null, 2);
                } else {
                    document.getElementById('ai-section').style.display = 'none';
                }

            } catch (err) {
                console.error(err);
                alert('Analysis failed. Check logs.');
            } finally {
                btnText.style.display = 'block';
                btnLoader.style.display = 'none';
                processBtn.disabled = false;
            }
        };

        // Drag and drop logic
        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.style.background = 'rgba(99, 102, 241, 0.1)'; };
        dropZone.ondragleave = () => { dropZone.style.background = 'transparent'; };
        dropZone.ondrop = (e) => {
            e.preventDefault();
            dropZone.style.background = 'transparent';
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                fileInput.onchange({target: fileInput});
            }
        };

        function syntaxHighlight(json) {
            json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
                var cls = 'json-number';
                if (/^"/.test(match)) {
                    if (/:$/.test(match)) {
                        cls = 'json-key';
                    } else {
                        cls = 'json-string';
                    }
                } else if (/true|false/.test(match)) {
                    cls = 'json-boolean';
                } else if (/null/.test(match)) {
                    cls = 'json-null';
                }
                return '<span class="' + cls + '">' + match + '</span>';
            });
        }
        
        function copyJson() {
            if (window.currentAiJson) {
                navigator.clipboard.writeText(window.currentAiJson).then(() => {
                    const btn = document.querySelector('.copy-btn');
                    const original = btn.innerText;
                    btn.innerText = 'Copied!';
                    setTimeout(() => btn.innerText = original, 2000);
                });
            }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
