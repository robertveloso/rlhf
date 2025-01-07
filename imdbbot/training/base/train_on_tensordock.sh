#!/bin/bash

# Check required environment variables
if [ -z "$TENSORDOCK_HOST" ] || [ -z "$TENSORDOCK_USER" ] || [ -z "$TENSORDOCK_PASSWORD" ] || [ -z "$TENSORDOCK_PORT" ]; then
    echo "Error: TensorDock credentials not set"
    echo "Please set the following environment variables:"
    echo "export TENSORDOCK_HOST=your-instance-ip"
    echo "export TENSORDOCK_USER=your-instance-user"
    echo "export TENSORDOCK_PASSWORD=your-instance-password"
    echo "export TENSORDOCK_PORT=your-ssh-port"
    exit 1
fi

if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: HUGGINGFACE_TOKEN environment variable is not set"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Then run: export HUGGINGFACE_TOKEN=your-token"
    exit 1
fi

# Install required Python packages
pip install paramiko
# pip freeze > requirements.txt

# Run deployment script
python deploy_to_tensordock.py