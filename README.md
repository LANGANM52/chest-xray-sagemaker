# Chest X-Ray Disease Detection with AWS SageMaker MLOps

A production-ready medical image classification system demonstrating end-to-end MLOps practices using AWS SageMaker, including model training, deployment, monitoring, and CI/CD automation.

## 🎯 Project Overview

This project implements a deep learning system for detecting diseases in chest X-ray images using the NIH Chest X-Ray dataset. It showcases:

- **Multi-label classification** of 14 disease conditions
- **SageMaker Training Jobs** with spot instances for cost optimization
- **SageMaker Serverless Inference** for scalable deployment
- **Experiment tracking** and model versioning
- **Production monitoring** with Prometheus and CloudWatch
- **CI/CD pipeline** with GitHub Actions
- **HIPAA-aware architecture** considerations

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Data Pipeline  │────▶│  Training Jobs   │────▶│  Model Registry │
│   (S3 + ETL)    │     │  (Spot Instance) │     │   (Versioning)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Monitoring    │◀────│  Serverless      │◀────│   FastAPI       │
│  (Prometheus)   │     │  Endpoint        │     │   Wrapper       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## 📊 Dataset

**NIH Chest X-Ray Dataset**
- 112,120 frontal-view X-ray images
- 30,805 unique patients
- 14 disease labels (multi-label classification)
- Publicly available, no PHI

Disease categories:
- Atelectasis, Cardiomegaly, Effusion, Infiltration
- Mass, Nodule, Pneumonia, Pneumothorax
- Consolidation, Edema, Emphysema, Fibrosis
- Pleural Thickening, Hernia

## 🚀 Quick Start

### Prerequisites
- AWS Account with SageMaker access
- Python 3.9+
- AWS CLI configured
- ~$50 budget for training and inference

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd chest-xray-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
aws configure
```

### Training

```bash
# Download and prepare dataset
python src/training/prepare_data.py

# Train model with SageMaker
python src/training/train_sagemaker.py --instance-type ml.g4dn.xlarge --use-spot
```

### Deployment

```bash
# Deploy to serverless endpoint
python src/inference/deploy.py --model-name chest-xray-v1

# Test inference
python src/inference/test_endpoint.py --image-path data/sample.jpg
```

## 🛠️ Technology Stack

**ML & Training:**
- PyTorch / TensorFlow
- SageMaker Training Jobs
- SageMaker Experiments
- DenseNet121 (pre-trained backbone)

**Deployment:**
- SageMaker Serverless Inference
- FastAPI
- Docker

**Monitoring & Observability:**
- SageMaker Model Monitor
- Prometheus
- CloudWatch
- JSON structured logging

**Infrastructure & CI/CD:**
- AWS CDK / CloudFormation
- GitHub Actions
- S3, IAM, VPC

## 💰 Cost Optimization Strategies

1. **Spot Instances**: 70% savings on training
2. **Serverless Inference**: Pay per request, no idle costs
3. **S3 Lifecycle Policies**: Automatic data archival
4. **Training Job Checkpointing**: Resume from failures
5. **Model Compression**: Reduce inference costs

**Estimated Costs:**
- Training: $10-20 (3-4 experiments)
- Inference: $5-10 (testing)
- Storage: $2-3/month
- **Total**: ~$30-50

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Average AUC-ROC | TBD |
| Accuracy | TBD |
| Inference Latency | TBD ms |
| Cost per 1K Predictions | TBD |

## 🔒 Healthcare Compliance

- Data encryption at rest and in transit
- VPC isolation for endpoints
- CloudTrail audit logging
- No PHI in logs or metrics
- Model explainability with GradCAM

## 📁 Project Structure

```
chest-xray-classifier/
├── src/
│   ├── training/          # Training scripts and data preparation
│   ├── inference/         # Deployment and inference code
│   └── monitoring/        # Metrics and monitoring setup
├── infrastructure/        # IaC (CDK/CloudFormation)
├── notebooks/            # Jupyter notebooks for exploration
├── tests/                # Unit and integration tests
├── docs/                 # Additional documentation
└── .github/workflows/    # CI/CD pipelines
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/ --aws-profile default
```

## 📚 Documentation

- [Architecture Deep Dive](docs/architecture.md)
- [Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)
- [Monitoring Setup](docs/monitoring.md)
- [Cost Analysis](docs/cost_analysis.md)

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome via issues.
