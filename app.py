"""
FastAPI web interface for Chest X-Ray disease detection.
Run with: uvicorn app:app --reload
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import boto3
import json
import io
from PIL import Image
import base64
from typing import Dict, List
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chest X-Ray Disease Detection",
    description="AI-powered chest X-ray analysis using AWS SageMaker",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
ENDPOINT_NAME = "chest-xray-serverless-2025-11-07-16-47"  # Update this!
REGION = "us-east-1"

# AWS SageMaker Runtime client
runtime = boto3.client('sagemaker-runtime', region_name=REGION)

# Disease labels
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chest X-Ray Disease Detection</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 30px;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 60px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8f9ff;
        }
        
        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f2ff;
            transform: translateY(-2px);
        }
        
        .upload-area.dragging {
            border-color: #764ba2;
            background: #e8ebff;
        }
        
        .upload-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        input[type="file"] {
            display: none;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 50px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .preview-container {
            margin: 30px 0;
            text-align: center;
        }
        
        .preview-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .results {
            margin-top: 30px;
        }
        
        .result-item {
            display: flex;
            align-items: center;
            margin: 15px 0;
            padding: 15px;
            background: #f8f9ff;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        .result-item:hover {
            background: #f0f2ff;
            transform: translateX(5px);
        }
        
        .result-label {
            flex: 0 0 200px;
            font-weight: 600;
            color: #667eea;
        }
        
        .result-bar-container {
            flex: 1;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 0 15px;
        }
        
        .result-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            padding: 0 10px;
            color: white;
            font-weight: 600;
        }
        
        .result-value {
            flex: 0 0 80px;
            text-align: right;
            font-weight: 600;
            font-size: 1.1em;
        }
        
        .confidence-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            margin-left: 10px;
        }
        
        .confidence-high {
            background: #4caf50;
            color: white;
        }
        
        .confidence-medium {
            background: #ff9800;
            color: white;
        }
        
        .confidence-low {
            background: #9e9e9e;
            color: white;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 30px 0;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            opacity: 0.9;
            font-size: 0.9em;
        }
        
        .error {
            background: #f44336;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
        }
        
        .error.active {
            display: block;
        }
        
        .warning {
            background: #fff3cd;
            border: 1px solid #ffc107;
            color: #856404;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        footer {
            text-align: center;
            color: white;
            margin-top: 50px;
            opacity: 0.8;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏥 Chest X-Ray AI Analyzer</h1>
            <p class="subtitle">Powered by AWS SageMaker | 14 Disease Categories</p>
        </header>
        
        <div class="card">
            <div class="warning">
                <strong>⚠️ Demo Purpose Only:</strong> This is a portfolio project for demonstration. 
                Not for actual medical diagnosis. Always consult healthcare professionals.
            </div>
            
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📁</div>
                <h2>Drop X-Ray Image Here</h2>
                <p>or click to browse</p>
                <p style="margin-top: 10px; opacity: 0.7;">Supports: JPG, PNG, JPEG</p>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            
            <div class="preview-container" id="previewContainer" style="display: none;">
                <h3>Selected Image:</h3>
                <img id="preview" class="preview-image" alt="Preview">
                <div style="margin-top: 20px;">
                    <button class="btn" id="analyzeBtn">🔍 Analyze X-Ray</button>
                    <button class="btn" id="clearBtn" style="background: #6c757d;">Clear</button>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 20px;">Analyzing X-Ray...</p>
            </div>
            
            <div class="error" id="error"></div>
        </div>
        
        <div class="card" id="resultsCard" style="display: none;">
            <h2>📊 Analysis Results</h2>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value" id="latency">--</div>
                    <div class="stat-label">Inference Time (ms)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="topDisease">--</div>
                    <div class="stat-label">Top Prediction</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="confidence">--</div>
                    <div class="stat-label">Confidence</div>
                </div>
            </div>
            
            <div class="results" id="results"></div>
        </div>
        
        <footer>
            <p>Built with AWS SageMaker, PyTorch & FastAPI</p>
            <p>© 2025 - Portfolio Project</p>
        </footer>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const preview = document.getElementById('preview');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const clearBtn = document.getElementById('clearBtn');
        const loading = document.getElementById('loading');
        const resultsCard = document.getElementById('resultsCard');
        const results = document.getElementById('results');
        const errorDiv = document.getElementById('error');
        
        let selectedFile = null;
        
        // Upload area click
        uploadArea.addEventListener('click', () => fileInput.click());
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragging');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragging');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragging');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleFile(file);
            }
        });
        
        // File input change
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) handleFile(file);
        });
        
        function handleFile(file) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                previewContainer.style.display = 'block';
                resultsCard.style.display = 'none';
                errorDiv.classList.remove('active');
            };
            reader.readAsDataURL(file);
        }
        
        // Clear button
        clearBtn.addEventListener('click', () => {
            selectedFile = null;
            fileInput.value = '';
            previewContainer.style.display = 'none';
            resultsCard.style.display = 'none';
            errorDiv.classList.remove('active');
        });
        
        // Analyze button
        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            loading.classList.add('active');
            analyzeBtn.disabled = true;
            errorDiv.classList.remove('active');
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            const startTime = performance.now();
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const endTime = performance.now();
                const latency = Math.round(endTime - startTime);
                
                if (!response.ok) {
                    throw new Error(await response.text());
                }
                
                const data = await response.json();
                displayResults(data, latency);
                
            } catch (error) {
                console.error('Error:', error);
                errorDiv.textContent = '❌ Error: ' + error.message;
                errorDiv.classList.add('active');
            } finally {
                loading.classList.remove('active');
                analyzeBtn.disabled = false;
            }
        });
        
        function displayResults(data, latency) {
            // Update stats
            document.getElementById('latency').textContent = latency;
            document.getElementById('topDisease').textContent = data.predictions[0].disease;
            document.getElementById('confidence').textContent = 
                (data.predictions[0].probability * 100).toFixed(1) + '%';
            
            // Show top 10 predictions
            results.innerHTML = '';
            data.predictions.slice(0, 10).forEach((pred, index) => {
                const percentage = (pred.probability * 100).toFixed(1);
                const confidenceClass = pred.confidence === 'high' ? 'confidence-high' :
                                       pred.confidence === 'medium' ? 'confidence-medium' : 'confidence-low';
                
                const item = document.createElement('div');
                item.className = 'result-item';
                item.style.animationDelay = `${index * 0.1}s`;
                item.innerHTML = `
                    <div class="result-label">${index + 1}. ${pred.disease}</div>
                    <div class="result-bar-container">
                        <div class="result-bar" style="width: ${percentage}%">
                            ${percentage}%
                        </div>
                    </div>
                    <div class="result-value">
                        <span class="confidence-badge ${confidenceClass}">
                            ${pred.confidence.toUpperCase()}
                        </span>
                    </div>
                `;
                results.appendChild(item);
            });
            
            resultsCard.style.display = 'block';
            resultsCard.scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict diseases from uploaded X-ray image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to expected size
        image = image.resize((224, 224))
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Invoke SageMaker endpoint
        logger.info(f"Invoking endpoint: {ENDPOINT_NAME}")
        
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/x-image',
            Body=img_byte_arr
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        logger.info("Prediction successful")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/endpoint/status")
async def endpoint_status():
    """Check if SageMaker endpoint is available"""
    try:
        sagemaker = boto3.client('sagemaker', region_name=REGION)
        response = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
        
        return {
            "endpoint_name": ENDPOINT_NAME,
            "status": response['EndpointStatus'],
            "creation_time": response['CreationTime'].isoformat()
        }
    except Exception as e:
        return {
            "endpoint_name": ENDPOINT_NAME,
            "status": "NotFound",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
