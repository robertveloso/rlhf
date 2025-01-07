from lambdacloud import LambdaCloud
import os
import time

def deploy_training():
    # Initialize Lambda Labs client
    client = LambdaCloud(api_key=os.environ.get('LAMBDA_API_KEY'))

    # Define instance configuration
    instance_config = {
        'region_name': 'us-east-1',  # or your preferred region
        'instance_type': 'gpu_1x_a10',  # or other GPU types
        'ssh_key_names': ['your-ssh-key'],  # your SSH key name
        'file_system_names': [],
        'quantity': 1
    }

    print("Starting Lambda Labs instance...")
    instances = client.create_instances(**instance_config)
    instance = instances[0]

    # Wait for instance to be ready
    while instance.status != 'active':
        time.sleep(10)
        instance.refresh()

    print(f"Instance ready: {instance.ip}")

    # Upload training files
    print("Uploading training files...")
    instance.upload_files([
        'train_model.py',
        'requirements.txt',
        '.env'
    ])

    # Run training
    print("Starting training...")
    instance.run_command('''
        # Setup environment
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt

        # Login to Hugging Face
        python -c "from huggingface_hub.cli.cli import login; login('$HUGGINGFACE_TOKEN')"

        # Run training
        python train_model.py
    ''')

    print("Training started on Lambda Labs!")
    print(f"Monitor your instance at: https://cloud.lambdalabs.com/instances/{instance.id}")

    return instance.id

if __name__ == "__main__":
    deploy_training()