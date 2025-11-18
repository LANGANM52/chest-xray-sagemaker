# System Architecture

## Overview

This document describes the architecture of the Chest X-Ray Disease Detection system, a production-ready ML system built on AWS SageMaker with MLOps best practices.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│                  (API / Web Application)                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API Gateway / Load Balancer                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               SageMaker Serverless Endpoint                      │
│                                                                   │
│  ┌──────────────────┐      ┌──────────────────┐                │
│  │   Model A        │      │   Model B        │                │
│  │   (DenseNet121)  │      │   (A/B Testing)  │                │
│  └──────────────────┘      └──────────────────┘                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Monitoring & Logging                         │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  CloudWatch  │  │  Prometheus  │  │ Model Monitor│         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Pipeline

**Purpose**: Prepare and manage training data

**Components**:
- **S3 Storage**: Raw images and metadata
- **Data Preparation**: ETL scripts for data cleaning and splitting
- **Feature Store** (Optional): Managed feature storage

**Data Flow**:
1. NIH Chest X-Ray dataset downloaded
2. Metadata processed and cleaned
3. Train/val/test splits created
4. Data uploaded to S3 in optimized format

### 2. Training Pipeline

**Purpose**: Train and experiment with models

**Components**:
- **SageMaker Training Jobs**: Managed training infrastructure
- **Spot Instances**: Cost-optimized compute (70% savings)
- **Experiment Tracking**: SageMaker Experiments for versioning
- **Model Registry**: Version control for trained models

**Training Flow**:
1. Training job launched via SageMaker SDK
2. Training data pulled from S3
3. Model trained on GPU instances (ml.g4dn.xlarge)
4. Checkpoints saved to S3
5. Model artifacts registered in Model Registry
6. Metrics logged to SageMaker Experiments

**Cost Optimization**:
- Spot instances for 70% cost reduction
- Automatic checkpoint/resume on interruption
- Instance right-sizing based on workload

### 3. Model Architecture

**Base Model**: DenseNet121 (pre-trained on ImageNet)

**Modifications**:
- Custom classifier head for multi-label classification
- 14 output neurons (one per disease)
- Dropout layers for regularization

**Input**: 224x224 RGB chest X-ray image

**Output**: 14 probability scores (0-1) for each disease

```python
Input (224x224x3)
    ↓
DenseNet121 Backbone (pre-trained)
    ↓
Global Average Pooling
    ↓
FC Layer (1024 → 512) + ReLU + Dropout(0.3)
    ↓
FC Layer (512 → 14) + Sigmoid
    ↓
Output (14 probabilities)
```

### 4. Inference Pipeline

**Purpose**: Serve predictions in production

**Deployment Options**:

#### A. Serverless Inference (Recommended)
- **Use Case**: Variable traffic, cost optimization
- **Configuration**:
  - Memory: 4096 MB
  - Max Concurrency: 10
  - Cold Start: ~10-30 seconds
  - Warm Latency: ~100-500ms
- **Cost**: Pay per millisecond of inference time
- **No idle costs** when not in use

#### B. Real-Time Endpoint
- **Use Case**: Consistent low-latency requirements
- **Configuration**:
  - Instance: ml.t2.medium or ml.g4dn.xlarge
  - Min instances: 1
  - Auto-scaling: Based on invocation metrics
- **Cost**: Hourly instance charges (even when idle)

### 5. Monitoring System

**Metrics Collection**:

1. **Application Metrics** (Prometheus):
   - Prediction count
   - Latency distribution
   - Confidence scores
   - Error rates

2. **Infrastructure Metrics** (CloudWatch):
   - Endpoint invocations
   - Model latency
   - Memory utilization
   - Throttling events

3. **Model Performance**:
   - SageMaker Model Monitor for drift detection
   - Ground truth labeling for accuracy tracking
   - A/B test results

**Alerting**:
- CloudWatch Alarms for endpoint health
- SNS notifications for critical issues
- PagerDuty integration (optional)

### 6. CI/CD Pipeline

**Stages**:

1. **Code Quality**:
   - Linting (flake8)
   - Formatting (black)
   - Type checking (mypy)
   - Security scanning (bandit)

2. **Testing**:
   - Unit tests
   - Integration tests
   - Model performance tests

3. **Staging Deployment**:
   - Deploy to staging endpoint
   - Run smoke tests
   - Performance validation

4. **Production Deployment**:
   - Manual approval required
   - Blue-green deployment
   - Gradual traffic shifting
   - Rollback capability

## Security Architecture

### Data Security
- **Encryption at Rest**: S3 server-side encryption (SSE-S3)
- **Encryption in Transit**: TLS 1.2+ for all API calls
- **Access Control**: IAM roles with least privilege
- **No PHI**: Dataset is de-identified

### Network Security
- **VPC Isolation**: Endpoints in private subnets
- **Security Groups**: Restrict inbound/outbound traffic
- **PrivateLink**: Internal AWS service communication

### Compliance
- **HIPAA-Ready**: Architecture can be HIPAA compliant
- **Audit Logging**: CloudTrail for all API calls
- **Data Retention**: Configurable lifecycle policies

## Scalability

### Horizontal Scaling
- Serverless endpoint automatically scales
- Model Monitor can handle increased load
- S3 scales automatically

### Vertical Scaling
- Instance types can be upgraded
- Memory configuration adjustable
- GPU instances available for complex models

## Disaster Recovery

### Backup Strategy
- **Model Artifacts**: Versioned in S3 with lifecycle rules
- **Training Data**: Replicated across AZs
- **Checkpoints**: Automatic during training

### Recovery Time Objectives
- **RTO**: < 1 hour (redeploy endpoint)
- **RPO**: 0 (models are stateless)

## Cost Analysis

### Monthly Cost Estimate (Development)

| Component | Cost |
|-----------|------|
| Training (3-4 jobs/month) | $15-20 |
| Serverless Inference (1000 requests) | $0.20 |
| S3 Storage (50GB) | $1.15 |
| CloudWatch Logs | $2-5 |
| **Total** | **~$20-30/month** |

### Cost Optimization Strategies
1. Use spot instances (70% savings)
2. Serverless inference (no idle costs)
3. S3 lifecycle policies (archive old data)
4. Right-size instances
5. Delete unused endpoints

## Future Enhancements

1. **Multi-Model Endpoints**: Serve multiple models from one endpoint
2. **Feature Store**: Centralized feature management
3. **AutoML**: Automated hyperparameter tuning
4. **Explainability**: Add SHAP/LIME for model interpretation
5. **Edge Deployment**: Deploy to edge devices with SageMaker Neo
6. **Federated Learning**: Train on distributed datasets

## References

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [NIH Chest X-Ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [DenseNet Paper](https://arxiv.org/abs/1608.06993)
