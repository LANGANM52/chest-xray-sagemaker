"""
Data preparation script for NIH Chest X-Ray dataset.
Downloads, processes, and uploads data to S3 for SageMaker training.
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import boto3
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import click

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChestXRayDataPreparer:
    """Handles downloading and preparing the NIH Chest X-Ray dataset"""
    
    # NIH dataset URLs
    METADATA_URL = "https://nihcc.app.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz"
    
    # Image links (there are 12 parts)
    IMAGE_URLS = [
        "https://nihcc.app.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
        # Add more URLs as needed - for demo we'll work with subset
    ]
    
    DISEASE_LABELS = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia"
    ]
    
    def __init__(
        self, 
        data_dir: str = "./data",
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15
    ):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.metadata_file = self.data_dir / "Data_Entry_2017_v2020.csv"
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
    def download_metadata(self):
        """Download the metadata CSV file"""
        logger.info("Downloading metadata...")
        
        # For this demo, we'll create a sample metadata file
        # In production, you would download from NIH
        logger.info("Creating sample metadata for demonstration...")
        
        # Create sample data
        sample_data = {
            'Image Index': [f'00000{i:03d}_000.png' for i in range(1000)],
            'Finding Labels': np.random.choice(
                self.DISEASE_LABELS + ['No Finding'], 
                size=1000
            ),
            'Patient ID': np.random.randint(1, 100, size=1000),
            'Patient Age': np.random.randint(18, 90, size=1000),
            'Patient Gender': np.random.choice(['M', 'F'], size=1000),
            'View Position': np.random.choice(['PA', 'AP'], size=1000)
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(self.metadata_file, index=False)
        logger.info(f"Metadata saved to {self.metadata_file}")
        
        return df
    
    def process_metadata(self) -> pd.DataFrame:
        """Process metadata and create multi-label encodings"""
        logger.info("Processing metadata...")
        
        # Load metadata
        if not self.metadata_file.exists():
            df = self.download_metadata()
        else:
            df = pd.read_csv(self.metadata_file)
        
        logger.info(f"Loaded {len(df)} images")
        
        # Create binary labels for each disease
        for label in self.DISEASE_LABELS:
            df[label] = df['Finding Labels'].str.contains(label).astype(int)
        
        # Filter out images with no findings for balanced dataset
        df_with_findings = df[df['Finding Labels'] != 'No Finding']
        logger.info(f"Images with findings: {len(df_with_findings)}")
        
        return df_with_findings
    
    def create_splits(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        logger.info("Creating data splits...")
        
        # First split: train + val vs test
        train_val, test = train_test_split(
            df,
            test_size=self.test_split,
            random_state=42,
            stratify=df['Patient Gender']  # Stratify by gender
        )
        
        # Second split: train vs val
        val_size = self.val_split / (self.train_split + self.val_split)
        train, val = train_test_split(
            train_val,
            test_size=val_size,
            random_state=42,
            stratify=train_val['Patient Gender']
        )
        
        logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train, val, test
    
    def save_splits(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame
    ):
        """Save split datasets to CSV files"""
        logger.info("Saving splits...")
        
        splits_dir = self.data_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        train.to_csv(splits_dir / "train.csv", index=False)
        val.to_csv(splits_dir / "val.csv", index=False)
        test.to_csv(splits_dir / "test.csv", index=False)
        
        logger.info(f"Splits saved to {splits_dir}")
    
    def upload_to_s3(
        self,
        bucket_name: str,
        prefix: str = "data"
    ):
        """Upload processed data to S3"""
        logger.info(f"Uploading data to s3://{bucket_name}/{prefix}")
        
        s3_client = boto3.client('s3')
        
        # Upload splits
        splits_dir = self.data_dir / "splits"
        for file in splits_dir.glob("*.csv"):
            s3_key = f"{prefix}/splits/{file.name}"
            logger.info(f"Uploading {file.name}...")
            s3_client.upload_file(str(file), bucket_name, s3_key)
        
        logger.info("Upload complete!")
        
        return f"s3://{bucket_name}/{prefix}"
    
    def generate_statistics(self, df: pd.DataFrame):
        """Generate and log dataset statistics"""
        logger.info("\n=== Dataset Statistics ===")
        logger.info(f"Total images: {len(df)}")
        logger.info(f"Unique patients: {df['Patient ID'].nunique()}")
        logger.info(f"Age range: {df['Patient Age'].min()} - {df['Patient Age'].max()}")
        logger.info(f"Gender distribution:\n{df['Patient Gender'].value_counts()}")
        logger.info(f"\nDisease prevalence:")
        
        for label in self.DISEASE_LABELS:
            if label in df.columns:
                count = df[label].sum()
                pct = (count / len(df)) * 100
                logger.info(f"  {label}: {count} ({pct:.2f}%)")


@click.command()
@click.option('--data-dir', default='./data', help='Local data directory')
@click.option('--s3-bucket', default=None, help='S3 bucket for upload')
@click.option('--s3-prefix', default='data', help='S3 prefix')
@click.option('--skip-download', is_flag=True, help='Skip downloading data')
def main(data_dir, s3_bucket, s3_prefix, skip_download):
    """Prepare NIH Chest X-Ray dataset for SageMaker training"""
    
    preparer = ChestXRayDataPreparer(data_dir=data_dir)
    
    # Process metadata
    if not skip_download:
        df = preparer.download_metadata()
    
    df = preparer.process_metadata()
    
    # Generate statistics
    preparer.generate_statistics(df)
    
    # Create splits
    train, val, test = preparer.create_splits(df)
    
    # Save splits locally
    preparer.save_splits(train, val, test)
    
    # Upload to S3 if bucket specified
    if s3_bucket:
        s3_uri = preparer.upload_to_s3(s3_bucket, s3_prefix)
        logger.info(f"\nData uploaded to: {s3_uri}")
        logger.info("\nNext steps:")
        logger.info("1. Review the data splits")
        logger.info("2. Run training: python src/training/train_sagemaker.py")
    else:
        logger.info("\nData prepared locally!")
        logger.info("To upload to S3, run with --s3-bucket option")


if __name__ == "__main__":
    main()
