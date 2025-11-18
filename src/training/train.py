"""
PyTorch training script for chest X-ray classification.
This script will be executed by SageMaker Training Jobs.
"""
import argparse
import os
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChestXRayDataset(Dataset):
    """Custom dataset for chest X-ray images"""
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        image_dir: str,
        disease_labels: list,
        transform=None,
        is_training: bool = True
    ):
        self.data_df = data_df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.disease_labels = disease_labels
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        # Load image
        img_path = self.image_dir / row['Image Index']
        
        # For demo purposes, create a dummy image if file doesn't exist
        if not img_path.exists():
            image = Image.new('RGB', (224, 224), color='gray')
        else:
            image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        labels = torch.tensor([
            row[label] for label in self.disease_labels
        ], dtype=torch.float32)
        
        return image, labels


class ChestXRayClassifier(nn.Module):
    """Multi-label classification model using DenseNet121"""
    
    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super(ChestXRayClassifier, self).__init__()
        
        # Load pre-trained DenseNet121
        self.densenet = models.densenet121(pretrained=pretrained)
        
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


def get_transforms(image_size: int, is_training: bool = True):
    """Get data augmentation transforms"""
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """Train for one epoch"""
    
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return running_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Validate the model"""
    
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Store predictions
            predictions = torch.sigmoid(outputs)
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
    
    # Calculate metrics
    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)
    
    # Calculate AUC-ROC per class
    auc_scores = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:  # Need both classes
            auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
            auc_scores.append(auc)
    
    avg_auc = np.mean(auc_scores) if auc_scores else 0.0
    avg_loss = running_loss / len(val_loader)
    
    return avg_loss, avg_auc


def save_model(model: nn.Module, model_dir: str):
    """Save model for SageMaker"""
    
    model_path = Path(model_dir) / "model.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


def parse_args():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--num-classes', type=int, default=14)
    
    # SageMaker specific parameters
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data/splits'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', './data/splits'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    
    return parser.parse_args()


def main():
    """Main training function"""
    
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Disease labels
    disease_labels = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia"
    ]
    
    # Load data
    logger.info("Loading training data...")
    train_df = pd.read_csv(Path(args.train) / 'train.csv')
    val_df = pd.read_csv(Path(args.validation) / 'val.csv')
    
    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = ChestXRayDataset(
        train_df,
        image_dir='./data/images',  # Dummy for demo
        disease_labels=disease_labels,
        transform=get_transforms(args.image_size, is_training=True),
        is_training=True
    )
    
    val_dataset = ChestXRayDataset(
        val_df,
        image_dir='./data/images',
        disease_labels=disease_labels,
        transform=get_transforms(args.image_size, is_training=False),
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = ChestXRayClassifier(num_classes=args.num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_auc = 0.0
    metrics_history = []
    
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # Save metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auc': val_auc
        }
        metrics_history.append(metrics)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            save_model(model, args.model_dir)
            logger.info(f"New best model saved! AUC: {best_auc:.4f}")
    
    # Save final metrics
    metrics_path = Path(args.output_data_dir) / 'metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    logger.info(f"\nTraining complete! Best AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
