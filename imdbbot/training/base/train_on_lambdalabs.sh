#!/bin/bash

# Check if LAMBDA_API_KEY is set
if [ -z "$LAMBDA_API_KEY" ]; then
    echo "Error: LAMBDA_API_KEY environment variable is not set"
    echo "Get your API key from: https://cloud.lambdalabs.com/api-keys"
    echo "Then run: export LAMBDA_API_KEY=your-api-key"
    exit 1
fi

# Check if HUGGINGFACE_TOKEN is set
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: HUGGINGFACE_TOKEN environment variable is not set"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Then run: export HUGGINGFACE_TOKEN=your-token"
    exit 1
fi

# Run deployment script
python deploy_to_lambda.py