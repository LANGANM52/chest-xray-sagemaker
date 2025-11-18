# Cost Analysis & Optimization Guide

## Overview

This document provides a detailed breakdown of AWS costs for the Chest X-Ray Classification project and strategies to stay within the $100 budget.

## Cost Breakdown

### 1. Training Costs

#### SageMaker Training Jobs

**Without Spot Instances:**
```
Instance: ml.g4dn.xlarge
Price: ~$0.736/hour
Training time: ~1-2 hours per job
Cost per job: $0.74 - $1.47
3-4 training jobs: $2.22 - $5.88
```

**With Spot Instances (70% discount):**
```
Instance: ml.g4dn.xlarge (spot)
Price: ~$0.221/hour
Training time: ~1-2 hours per job
Cost per job: $0.22 - $0.44
3-4 training jobs: $0.66 - $1.76 ✅
```

**Recommendation**: Always use spot instances for training!

### 2. Inference Costs

#### Serverless Inference (Recommended)

```
Cost Structure:
- Compute: $0.20 per 1M inference milliseconds
- Memory: Based on configuration (4096 MB)

Example Calculations:
1000 predictions × 500ms each = 500,000 ms
Cost: 500,000 ÷ 1,000,000 × $0.20 = $0.10 ✅

10,000 predictions × 500ms each = 5,000,000 ms
Cost: 5,000,000 ÷ 1,000,000 × $0.20 = $1.00 ✅

Benefits:
- No idle costs
- Automatic scaling
- Only pay for actual usage
```

#### Real-Time Endpoint (For Comparison)

```
Instance: ml.t2.medium
Price: ~$0.065/hour
Monthly cost (24/7): ~$47/month ❌

Instance: ml.g4dn.xlarge
Price: ~$0.736/hour
Monthly cost (24/7): ~$531/month ❌❌

Note: Real-time endpoints charge even when idle!
```

**Recommendation**: Use serverless for development and low-volume production.

### 3. Storage Costs (S3)

```
Data Storage:
- Standard storage: $0.023 per GB/month
- Intelligent-Tiering: $0.023 per GB/month (auto-optimization)

Example:
50 GB dataset: 50 × $0.023 = $1.15/month ✅
100 GB dataset: 100 × $0.023 = $2.30/month ✅

Model Artifacts:
- ~500 MB per model: $0.012/month
- 5 versions: $0.06/month ✅
```

### 4. CloudWatch Costs

```
Log Storage:
- First 5 GB: Free
- Additional: $0.50 per GB

Metrics:
- First 10 custom metrics: Free
- Additional: $0.30 per metric

Estimated monthly: $2-5 ✅
```

### 5. Data Transfer Costs

```
Inbound: Free
Outbound:
- First 100 GB/month: Free
- Next 10 TB: $0.09 per GB

For this project: Minimal (mostly internal AWS traffic)
Estimated: < $1/month ✅
```

## Complete Budget Estimate

### Development Phase (1 month)

| Item | Cost |
|------|------|
| Training (4 jobs, spot) | $2-4 |
| Serverless Inference (testing, 5000 predictions) | $0.50 |
| S3 Storage (50 GB) | $1.15 |
| CloudWatch | $2-3 |
| Data Transfer | $0.50 |
| **Total** | **~$6-11** ✅ |

### Production Phase (1 month)

| Item | Cost |
|------|------|
| Training (2 jobs, spot) | $1-2 |
| Serverless Inference (20,000 predictions) | $2-4 |
| S3 Storage (75 GB) | $1.73 |
| CloudWatch | $3-5 |
| Model Monitor | $2-3 |
| **Total** | **~$10-17** ✅ |

### Full Project (3 months)

| Item | Cost |
|------|------|
| Initial Training & Experimentation | $10-15 |
| Development & Testing | $5-10 |
| Production Deployment & Monitoring | $15-25 |
| **Total** | **~$30-50** ✅ |

**Remaining Budget**: $50-70 for additional experiments!

## Cost Optimization Strategies

### 1. Training Optimization

✅ **Use Spot Instances** (70% savings)
```python
use_spot_instances=True
max_wait_time=86400  # 24 hours
```

✅ **Enable Checkpointing**
```python
checkpoint_s3_uri='s3://bucket/checkpoints'
checkpoint_local_path='/opt/ml/checkpoints'
```

✅ **Right-Size Instances**
- Start with ml.g4dn.xlarge
- Monitor GPU utilization
- Scale down if underutilized

✅ **Use Managed Warmup Pools**
```python
keep_alive_period_in_seconds=600  # 10 minutes
```

### 2. Inference Optimization

✅ **Serverless for Variable Traffic**
```python
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=4096,  # Start here
    max_concurrency=10       # Adjust based on load
)
```

✅ **Batch Predictions**
- Group predictions when possible
- Reduce per-request overhead

✅ **Model Compression**
- Quantization (INT8 instead of FP32)
- Pruning unused connections
- Knowledge distillation

### 3. Storage Optimization

✅ **S3 Lifecycle Policies**
```json
{
  "Rules": [{
    "Id": "archive-old-models",
    "Status": "Enabled",
    "Transitions": [{
      "Days": 30,
      "StorageClass": "INTELLIGENT_TIERING"
    }]
  }]
}
```

✅ **Delete Unused Resources**
```bash
# List and delete old models
aws s3 ls s3://bucket/models/
aws s3 rm s3://bucket/models/old-model.tar.gz
```

### 4. Monitoring Optimization

✅ **Reduce Log Retention**
```python
retention_in_days=7  # Instead of default 30
```

✅ **Sample Metrics**
```python
data_capture_percentage=10  # Sample 10% instead of 100%
```

### 5. Development Best Practices

✅ **Clean Up After Each Session**
```bash
# Delete endpoint after testing
python src/inference/deploy.py --delete-endpoint test-endpoint
```

✅ **Use Local Development**
- Test code locally before SageMaker
- Use smaller datasets for debugging

✅ **Set Budget Alarms**
```bash
aws budgets create-budget \
  --account-id 123456789 \
  --budget file://budget.json
```

## Cost Monitoring Commands

### Check Current Costs
```bash
# Get cost explorer data
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost
```

### List Active Resources
```bash
# List training jobs
aws sagemaker list-training-jobs

# List endpoints
aws sagemaker list-endpoints

# List models
aws sagemaker list-models
```

### Delete Resources
```bash
# Stop training job
aws sagemaker stop-training-job --training-job-name job-name

# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name endpoint-name

# Delete model
aws sagemaker delete-model --model-name model-name
```

## Budget Alert Setup

### Create CloudWatch Billing Alarm

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_alarm(
    AlarmName='BudgetAlert-$50',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='EstimatedCharges',
    Namespace='AWS/Billing',
    Period=21600,  # 6 hours
    Statistic='Maximum',
    Threshold=50.0,
    ActionsEnabled=True,
    AlarmActions=['arn:aws:sns:region:account:topic'],
    AlarmDescription='Alert when bill exceeds $50'
)
```

## Cost Comparison: Alternatives

### vs. On-Premise Training
```
On-Premise (GPU Workstation):
- Hardware: $2,000-5,000 upfront
- Electricity: ~$50-100/month
- Maintenance: Time + $$

SageMaker (Spot):
- No upfront cost
- Pay as you go: ~$2-4 per training job
- Zero maintenance

Winner: SageMaker ✅ (especially for learning/portfolio)
```

### vs. Other Cloud Providers

```
AWS SageMaker (Spot):
- Training: $0.221/hour
- Serverless: $0.20/1M ms
- Total: ~$30-50 for project

GCP Vertex AI:
- Training: ~$0.30/hour (preemptible)
- Endpoint: Similar costs
- Total: Comparable

Azure ML:
- Training: ~$0.35/hour (low priority)
- Endpoint: Similar costs
- Total: Slightly higher

Winner: AWS ✅ (best spot pricing + serverless options)
```

## Emergency Cost Control

If you're approaching budget limits:

1. **Immediately**:
   - Stop all training jobs
   - Delete all endpoints
   - List and review all resources

2. **Within 24 hours**:
   - Set up budget alarms
   - Review CloudWatch logs
   - Clean up unused S3 objects

3. **Going Forward**:
   - Use smaller datasets for testing
   - Reduce training frequency
   - Use lower memory serverless config

## Real Project Cost Example

**Scenario**: Building this portfolio project

```
Week 1: Setup & Initial Training
- 5 training experiments: $2-3
- Testing endpoints: $0.50
- Storage: $0.30
Total: $2.80

Week 2: Model Refinement
- 3 training jobs: $1-2
- Extended testing: $1
- Storage: $0.30
Total: $2.30

Week 3: Deployment & Documentation
- 1 final training: $0.50
- Production testing: $2
- Storage: $0.30
Total: $2.80

Week 4: Portfolio Preparation
- Minimal compute: $0
- Storage: $0.30
- Monitoring: $1
Total: $1.30

GRAND TOTAL: ~$9.20 ✅✅✅

Budget remaining: $90.80 for future experiments!
```

## Conclusion

This project is designed to be **extremely cost-effective**. By following the strategies above, you can:

- Complete the entire project for **$30-50**
- Stay well within the **$100 budget**
- Have budget remaining for **additional experiments**

The key is using **spot instances for training** and **serverless inference** for deployment!
