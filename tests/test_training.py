"""
Unit tests for training module.
"""
import pytest
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training.train import ChestXRayDataset, ChestXRayClassifier, get_transforms


class TestChestXRayDataset:
    """Tests for ChestXRayDataset"""
    
    def test_dataset_initialization(self, sample_dataframe, tmp_path):
        """Test dataset can be initialized"""
        dataset = ChestXRayDataset(
            data_df=sample_dataframe,
            image_dir=str(tmp_path),
            disease_labels=['Pneumonia', 'Cardiomegaly'],
            transform=None
        )
        
        assert len(dataset) == len(sample_dataframe)
    
    def test_dataset_getitem(self, sample_dataframe, tmp_path):
        """Test getting an item from dataset"""
        # Create dummy image
        img_path = tmp_path / '00000001_000.png'
        Image.new('RGB', (224, 224)).save(img_path)
        
        dataset = ChestXRayDataset(
            data_df=sample_dataframe,
            image_dir=str(tmp_path),
            disease_labels=['Pneumonia', 'Cardiomegaly'],
            transform=get_transforms(224, is_training=False)
        )
        
        image, labels = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(labels, torch.Tensor)
        assert len(labels) == 2


class TestChestXRayClassifier:
    """Tests for ChestXRayClassifier model"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = ChestXRayClassifier(num_classes=14)
        assert model is not None
    
    def test_model_forward_pass(self):
        """Test forward pass through model"""
        model = ChestXRayClassifier(num_classes=14)
        model.eval()
        
        # Create dummy input
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 14)
    
    def test_model_output_range(self):
        """Test model outputs are valid probabilities after sigmoid"""
        model = ChestXRayClassifier(num_classes=14)
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
            probs = torch.sigmoid(output)
        
        assert torch.all(probs >= 0) and torch.all(probs <= 1)


class TestTransforms:
    """Tests for data transforms"""
    
    def test_training_transforms(self):
        """Test training transforms work"""
        transform = get_transforms(224, is_training=True)
        
        # Create dummy image
        img = Image.new('RGB', (512, 512))
        
        transformed = transform(img)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 224, 224)
    
    def test_validation_transforms(self):
        """Test validation transforms work"""
        transform = get_transforms(224, is_training=False)
        
        img = Image.new('RGB', (512, 512))
        transformed = transform(img)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 224, 224)


# Fixtures
@pytest.fixture
def sample_dataframe():
    """Create sample dataframe for testing"""
    import pandas as pd
    
    data = {
        'Image Index': ['00000001_000.png', '00000002_000.png'],
        'Pneumonia': [1, 0],
        'Cardiomegaly': [0, 1]
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_image():
    """Create sample image for testing"""
    return Image.new('RGB', (224, 224), color='gray')


# Integration tests
class TestTrainingPipeline:
    """Integration tests for training pipeline"""
    
    def test_full_training_step(self, sample_dataframe, tmp_path):
        """Test a single training step works end-to-end"""
        # This would test the full training loop
        # Simplified version here
        
        model = ChestXRayClassifier(num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Create dummy batch
        images = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        
        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
