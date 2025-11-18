"""
Inference handler for SageMaker endpoint.
Handles model loading and prediction requests.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChestXRayClassifier(nn.Module):
    """Multi-label classification model using DenseNet121"""
    
    def __init__(self, num_classes: int = 14):
        super(ChestXRayClassifier, self).__init__()
        
        # Load pre-trained DenseNet121
        self.densenet = models.densenet121(pretrained=False)
        
        # Replace classifier
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.densenet(x)


# Disease labels
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]


def model_fn(model_dir):
    """
    Load the PyTorch model from the `model_dir` directory.
    Called once when the endpoint is initialized.
    """
    logger.info("Loading model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChestXRayClassifier(num_classes=14)
    
    # Load model weights
    model_path = f"{model_dir}/model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    return model


def input_fn(request_body, content_type='application/x-image'):
    """
    Deserialize and prepare the prediction input.
    """
    logger.info(f"Processing input with content type: {content_type}")
    
    if content_type == 'application/x-image':
        # Image bytes
        image = Image.open(io.BytesIO(request_body)).convert('RGB')
        return image
    elif content_type == 'application/json':
        # Base64 encoded image
        input_data = json.loads(request_body)
        if 'image' in input_data:
            import base64
            image_bytes = base64.b64decode(input_data['image'])
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return image
    
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_object, model):
    """
    Apply model to the incoming request.
    """
    logger.info("Running prediction...")
    
    device = next(model.parameters()).device
    
    # Preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Transform image
    image_tensor = transform(input_object).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
    
    return probabilities


def output_fn(prediction, content_type='application/json'):
    """
    Serialize the prediction result.
    """
    logger.info("Formatting output...")
    
    # Create results dictionary
    results = {
        'predictions': [],
        'metadata': {
            'model_version': '1.0.0',
            'num_classes': len(DISEASE_LABELS)
        }
    }
    
    # Add each disease prediction
    for label, prob in zip(DISEASE_LABELS, prediction):
        results['predictions'].append({
            'disease': label,
            'probability': float(prob),
            'confidence': 'high' if prob > 0.7 else 'medium' if prob > 0.4 else 'low'
        })
    
    # Sort by probability
    results['predictions'].sort(key=lambda x: x['probability'], reverse=True)
    
    if content_type == 'application/json':
        return json.dumps(results)
    
    raise ValueError(f"Unsupported content type: {content_type}")


# Optional: Add health check endpoint
def ping():
    """Health check endpoint"""
    return {"status": "healthy"}
