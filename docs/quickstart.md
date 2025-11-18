# Quick Start Guide

## 🚀 Get Up and Running in 30 Minutes

This guide will help you set up and deploy your chest X-ray classifier quickly.

## Prerequisites

- AWS Account with ~$50 in credits
- Python 3.9+
- Git
- Basic command line knowledge

## Step 1: Initial Setup (5 minutes)

```bash
# Clone the repository
git clone <your-repo-url>
cd chest-xray-classifier

# Run setup script
./setup.sh

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Step 2: Configure AWS (5 minutes)

```bash
# Configure AWS CLI
aws configure
# Enter your:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (us-east-1)
# - Output format (json)

# Verify configuration
aws sts get-caller-identity
```

## Step 3: Set Up IAM Role (5 minutes)

### Option A: Using AWS Console

1. Go to [IAM Console](https://console.aws.amazon.com/iam/)
2. Click "Roles" → "Create role"
3. Select "AWS service" → "SageMaker"
4. Choose "SageMaker - Execution"
5. Add policies:
   - AmazonSageMakerFullAccess
   - AmazonS3FullAccess (or create custom with least privilege)
6. Name it: `SageMakerExecutionRole`
7. Copy the ARN

### Option B: Using AWS CLI

```bash
# Create trust policy
cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name SageMakerExecutionRole \
  --assume-role-policy-document file://trust-policy.json

# Attach policies
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Get role ARN
aws iam get-role --role-name SageMakerExecutionRole --query 'Role.Arn'
```

## Step 4: Update Configuration (2 minutes)

Edit `.env` file:

```bash
# Update these values
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012  # Your account ID
SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerExecutionRole
S3_BUCKET=chest-xray-ml-<your-unique-suffix>
```

Create S3 bucket:

```bash
# Replace with your unique bucket name
aws s3 mb s3://chest-xray-ml-<your-unique-suffix> --region us-east-1
```

## Step 5: Prepare Data (3 minutes)

```bash
# Prepare sample dataset
python src/training/prepare_data.py \
  --s3-bucket chest-xray-ml-<your-suffix> \
  --s3-prefix data

# This creates sample data for demonstration
# In production, you'd download the full NIH dataset
```

## Step 6: Train Model (30-60 minutes)

```bash
# Launch training job with spot instances
python src/training/train_sagemaker.py \
  --instance-type ml.g4dn.xlarge \
  --use-spot \
  --num-epochs 10 \
  --bucket chest-xray-ml-<your-suffix>

# Output will show:
# - Training job name
# - AWS Console link
# - CloudWatch logs link
# - Estimated cost (~$2-4)
# - Estimated time (30-60 min)
```

**💡 Tip**: The job runs in the background. You can close your terminal!

### Monitor Training

**Option 1: AWS Console**
- Click the link provided in the output
- Watch metrics in real-time

**Option 2: Command Line**
```bash
# List training jobs
aws sagemaker list-training-jobs --sort-by CreationTime --sort-order Descending

# Check specific job
aws sagemaker describe-training-job --training-job-name <job-name>
```

**Option 3: Python**
```python
import sagemaker

# Get training job status
session = sagemaker.Session()
job = session.describe_training_job('<job-name>')
print(job['TrainingJobStatus'])
```

## Step 7: Deploy Model (5-10 minutes)

After training completes:

```bash
# Deploy to serverless endpoint
python src/inference/deploy.py \
  --deployment-type serverless \
  --memory-size 4096 \
  --max-concurrency 10

# Output shows:
# - Endpoint name
# - Configuration
# - Cost structure
```

## Step 8: Test Endpoint (2 minutes)

```bash
# Single prediction test
python src/inference/test_endpoint.py \
  --endpoint-name chest-xray-serverless-<timestamp>

# Load test
python src/inference/test_endpoint.py \
  --endpoint-name chest-xray-serverless-<timestamp> \
  --load-test \
  --num-requests 10
```

## Step 9: Clean Up (Important!)

```bash
# Delete endpoint to avoid charges
python src/inference/deploy.py \
  --delete-endpoint chest-xray-serverless-<timestamp>

# Or using AWS CLI
aws sagemaker delete-endpoint --endpoint-name <endpoint-name>
```

## 🎉 Success!

You now have:
- ✅ Trained ML model
- ✅ Deployed serverless endpoint
- ✅ Tested predictions
- ✅ Cost-optimized setup

**Total Cost**: ~$5-10 for everything!

## Next Steps

### For Your Resume/Portfolio

1. **Create Architecture Diagram**
   - Use draw.io or Lucidchart
   - Include all components
   - Add to README

2. **Document Results**
   ```bash
   # Create results directory
   mkdir -p docs/results
   
   # Add:
   # - Training metrics plots
   # - Confusion matrices
   # - Example predictions
   # - Performance benchmarks
   ```

3. **Write Blog Post**
   - Explain the problem
   - Show your solution
   - Discuss challenges
   - Share learnings

4. **Make GitHub Repo Public**
   ```bash
   # Update README with your info
   # Add screenshots
   # Include demo video (optional)
   
   git add .
   git commit -m "Initial commit: Chest X-Ray Classification System"
   git push origin main
   ```

5. **Add to LinkedIn**
   - Write project description
   - Highlight technical skills
   - Link to GitHub repo
   - Share metrics/results

### For Further Development

1. **Improve Model**
   ```bash
   # Experiment with hyperparameters
   python src/training/train_sagemaker.py \
     --learning-rate 0.0001 \
     --batch-size 64 \
     --num-epochs 20
   
   # Try different architectures
   # - ResNet50
   # - EfficientNet
   # - Vision Transformer
   ```

2. **Add Features**
   - Grad-CAM visualization
   - Model interpretability (SHAP)
   - A/B testing between models
   - Real-time monitoring dashboard
   - Mobile app integration

3. **Production Enhancements**
   - Add authentication
   - Implement rate limiting
   - Create web UI
   - Add model versioning
   - Set up automated retraining

## Troubleshooting

### Training Job Fails

```bash
# Check CloudWatch logs
aws logs tail /aws/sagemaker/TrainingJobs --follow

# Common issues:
# - Insufficient permissions → Check IAM role
# - Out of memory → Reduce batch size
# - Spot interruption → Job will auto-resume
```

### Deployment Issues

```bash
# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name <name>

# Common issues:
# - Model not found → Check S3 path
# - Role permissions → Update IAM policies
# - Instance quota → Request quota increase
```

### Cost Concerns

```bash
# Check current costs
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost

# Set up budget alert (do this first!)
# See docs/cost_analysis.md
```

## Useful Commands

```bash
# List all SageMaker resources
aws sagemaker list-training-jobs
aws sagemaker list-endpoints
aws sagemaker list-models

# Delete resources
aws sagemaker stop-training-job --training-job-name <name>
aws sagemaker delete-endpoint --endpoint-name <name>
aws sagemaker delete-model --model-name <name>

# Check S3 usage
aws s3 ls s3://your-bucket --recursive --summarize
```

## Resources

- [AWS SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [AWS Free Tier](https://aws.amazon.com/free/)
- [SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)
- [NIH Dataset Info](https://nihcc.app.box.com/v/ChestXray-NIHCC)

## Getting Help

- Check documentation in `docs/`
- Review CloudWatch logs
- AWS Support (if you have a plan)
- Stack Overflow (tag: amazon-sagemaker)

---

**Remember**: Always clean up resources after experimenting to avoid unnecessary charges!
