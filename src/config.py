"""
Configuration management for the Chest X-Ray Classification project.
"""
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class AWSConfig(BaseSettings):
    """AWS-specific configuration"""
    region: str = Field(default="us-east-1", env="AWS_REGION")
    account_id: Optional[str] = Field(default=None, env="AWS_ACCOUNT_ID")
    sagemaker_role: Optional[str] = Field(default=None, env="SAGEMAKER_ROLE")
    s3_bucket: str = Field(default="chest-xray-ml-bucket", env="S3_BUCKET")
    
    # S3 Paths
    s3_data_prefix: str = "data"
    s3_model_prefix: str = "models"
    s3_output_prefix: str = "output"


class TrainingConfig(BaseSettings):
    """Training job configuration"""
    # Model
    model_name: str = "densenet121"
    num_classes: int = 14
    image_size: int = 224
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 10
    early_stopping_patience: int = 3
    
    # SageMaker Training
    instance_type: str = "ml.g4dn.xlarge"
    instance_count: int = 1
    use_spot_instances: bool = True
    max_wait_time: int = 86400  # 24 hours
    max_run_time: int = 43200   # 12 hours
    volume_size: int = 50  # GB
    
    # Checkpointing
    checkpoint_s3_uri: Optional[str] = None
    checkpoint_local_path: str = "/opt/ml/checkpoints"
    
    # Experiment tracking
    experiment_name: str = "chest-xray-classification"
    

class InferenceConfig(BaseSettings):
    """Inference configuration"""
    # Serverless configuration
    serverless_memory: int = 4096  # MB (2048, 4096, or 6144)
    serverless_max_concurrency: int = 10
    
    # Model settings
    confidence_threshold: float = 0.5
    batch_inference: bool = False
    
    # API settings
    api_port: int = 8000
    api_workers: int = 2
    api_timeout: int = 60


class MonitoringConfig(BaseSettings):
    """Monitoring and logging configuration"""
    # Prometheus
    prometheus_port: int = 9090
    metrics_path: str = "/metrics"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Model monitoring
    monitoring_schedule_cron: str = "cron(0 0 * * ? *)"  # Daily
    data_capture_percentage: int = 100
    baseline_dataset_s3_uri: Optional[str] = None
    

class DataConfig(BaseSettings):
    """Data pipeline configuration"""
    # Dataset
    dataset_name: str = "nih-chest-xray"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Disease labels
    disease_labels: List[str] = [
        "Atelectasis",
        "Cardiomegaly", 
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia"
    ]
    
    # Data augmentation
    use_augmentation: bool = True
    horizontal_flip: bool = True
    rotation_range: int = 10
    zoom_range: float = 0.1
    
    # Local paths
    local_data_dir: str = "./data"
    local_images_dir: str = "./data/images"
    local_metadata_file: str = "./data/Data_Entry_2017_v2020.csv"


class ProjectConfig(BaseSettings):
    """Main project configuration"""
    project_name: str = "chest-xray-classifier"
    version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Sub-configs
    aws: AWSConfig = AWSConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    data: DataConfig = DataConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global config instance
config = ProjectConfig()
