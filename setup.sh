#!/bin/bash
# Setup script for Chest X-Ray Classification project

set -e  # Exit on error

echo "================================================"
echo "Chest X-Ray Classifier - Project Setup"
echo "================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo -e "${YELLOW}Warning: Python 3.9+ required. You have $python_version${NC}"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo -e "${GREEN}✓ Dependencies installed successfully!${NC}"

# Check AWS CLI
echo ""
echo "Checking AWS CLI..."
if command -v aws &> /dev/null; then
    echo -e "${GREEN}✓ AWS CLI is installed${NC}"
    aws --version
else
    echo -e "${YELLOW}⚠ AWS CLI not found. Please install it:${NC}"
    echo "  https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
fi

# Check AWS credentials
echo ""
echo "Checking AWS credentials..."
if aws sts get-caller-identity &> /dev/null; then
    echo -e "${GREEN}✓ AWS credentials configured${NC}"
    aws sts get-caller-identity
else
    echo -e "${YELLOW}⚠ AWS credentials not configured. Run:${NC}"
    echo "  aws configure"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    cat > .env << EOF
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=your-account-id
SAGEMAKER_ROLE=arn:aws:iam::your-account-id:role/SageMakerRole
S3_BUCKET=chest-xray-ml-bucket

# Environment
ENVIRONMENT=development
EOF
    echo -e "${YELLOW}⚠ Please update .env with your AWS details${NC}"
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/splits
mkdir -p data/images
mkdir -p models
mkdir -p logs

echo ""
echo "================================================"
echo -e "${GREEN}Setup complete!${NC}"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Configure AWS credentials (if not done):"
echo "   aws configure"
echo ""
echo "3. Update .env file with your AWS details"
echo ""
echo "4. Prepare the dataset:"
echo "   python src/training/prepare_data.py --s3-bucket your-bucket"
echo ""
echo "5. Train the model:"
echo "   python src/training/train_sagemaker.py --use-spot"
echo ""
echo "6. Deploy the model:"
echo "   python src/inference/deploy.py"
echo ""
echo "For more information, see README.md"
echo ""
