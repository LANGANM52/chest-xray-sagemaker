"""
Test script for SageMaker inference endpoint.
"""
import boto3
import json
import numpy as np
from PIL import Image
import io
import click
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EndpointTester:
    """Test SageMaker endpoints"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.runtime = boto3.client('sagemaker-runtime', region_name=region)
        self.region = region
    
    def create_test_image(self, size=(224, 224)):
        """Create a test chest X-ray-like image"""
        # Create grayscale image
        image = Image.new('L', size, color=128)
        
        # Add some random noise to simulate X-ray texture
        pixels = np.array(image)
        noise = np.random.normal(0, 20, pixels.shape)
        pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
        
        image = Image.fromarray(pixels).convert('RGB')
        return image
    
    def invoke_endpoint(
        self,
        endpoint_name: str,
        image: Image.Image,
        content_type: str = 'application/x-image'
    ):
        """Send prediction request to endpoint"""
        
        logger.info(f"Invoking endpoint: {endpoint_name}")
        
        # Convert image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        # Invoke endpoint
        start_time = time.time()
        
        response = self.runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            Body=image_bytes
        )
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        logger.info(f"✓ Inference completed in {inference_time:.2f}ms")
        
        return result, inference_time
    
    def run_load_test(
        self,
        endpoint_name: str,
        num_requests: int = 10,
        delay: float = 0.5
    ):
        """Run simple load test"""
        
        logger.info(f"\nRunning load test: {num_requests} requests")
        logger.info("-" * 60)
        
        times = []
        successes = 0
        failures = 0
        
        for i in range(num_requests):
            try:
                image = self.create_test_image()
                result, inference_time = self.invoke_endpoint(endpoint_name, image)
                
                times.append(inference_time)
                successes += 1
                
                logger.info(f"Request {i+1}/{num_requests}: {inference_time:.2f}ms")
                
                # Small delay between requests
                if delay > 0:
                    time.sleep(delay)
                
            except Exception as e:
                failures += 1
                logger.error(f"Request {i+1} failed: {str(e)}")
        
        # Calculate statistics
        if times:
            logger.info(f"\n{'='*60}")
            logger.info("Load Test Results")
            logger.info(f"{'='*60}")
            logger.info(f"Total requests: {num_requests}")
            logger.info(f"Successful: {successes}")
            logger.info(f"Failed: {failures}")
            logger.info(f"Success rate: {(successes/num_requests)*100:.1f}%")
            logger.info(f"\nLatency statistics:")
            logger.info(f"  Mean: {np.mean(times):.2f}ms")
            logger.info(f"  Median: {np.median(times):.2f}ms")
            logger.info(f"  Min: {np.min(times):.2f}ms")
            logger.info(f"  Max: {np.max(times):.2f}ms")
            logger.info(f"  P95: {np.percentile(times, 95):.2f}ms")
            logger.info(f"  P99: {np.percentile(times, 99):.2f}ms")
    
    def display_predictions(self, result: dict, top_k: int = 5):
        """Display top predictions"""
        
        logger.info(f"\n{'='*60}")
        logger.info("Prediction Results")
        logger.info(f"{'='*60}")
        
        predictions = result['predictions'][:top_k]
        
        for i, pred in enumerate(predictions, 1):
            disease = pred['disease']
            prob = pred['probability']
            confidence = pred['confidence']
            
            bar = '█' * int(prob * 20)
            logger.info(f"{i}. {disease:20s} {prob:.3f} {bar} ({confidence})")


@click.command()
@click.option('--endpoint-name', required=True, help='SageMaker endpoint name')
@click.option('--image-path', default=None, help='Path to test image')
@click.option('--load-test', is_flag=True, help='Run load test')
@click.option('--num-requests', default=10, help='Number of requests for load test')
@click.option('--region', default='us-east-1', help='AWS region')
def main(endpoint_name, image_path, load_test, num_requests, region):
    """Test SageMaker inference endpoint"""
    
    try:
        tester = EndpointTester(region=region)
        
        if load_test:
            # Run load test
            tester.run_load_test(endpoint_name, num_requests=num_requests)
        else:
            # Single prediction
            if image_path and Path(image_path).exists():
                image = Image.open(image_path).convert('RGB')
                logger.info(f"Loaded image: {image_path}")
            else:
                logger.info("No image provided, creating test image...")
                image = tester.create_test_image()
            
            # Invoke endpoint
            result, inference_time = tester.invoke_endpoint(endpoint_name, image)
            
            # Display results
            tester.display_predictions(result)
            
            logger.info(f"\nInference time: {inference_time:.2f}ms")
            logger.info(f"\n✓ Test complete!")
            
    except Exception as e:
        logger.error(f"Error testing endpoint: {str(e)}")
        raise


if __name__ == '__main__':
    main()
