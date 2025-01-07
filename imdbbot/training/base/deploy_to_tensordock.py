import paramiko
import os
from time import sleep

def deploy_training():
    # TensorDock SSH credentials (from their dashboard)
    hostname = os.environ.get('TENSORDOCK_HOST')
    username = os.environ.get('TENSORDOCK_USER', 'root')
    password = os.environ.get('TENSORDOCK_PASSWORD')
    port = int(os.environ.get('TENSORDOCK_PORT', '22'))

    print("Connecting to TensorDock instance...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"Preparing to open connection to TensorDock instance {hostname} on port {port} with username {username} and password {password}")
        ssh.connect(hostname, port, username, password)
        sftp = ssh.open_sftp()

        # Upload training files
        print("Uploading training files...")
        files_to_upload = [
            ('imdbbot/training/base/train_model.py', '/home/user/train_model.py'),
            ('requirements.txt', '/home/user/requirements.txt'),
            ('.env', '/home/user/.env')
        ]

        for local_path, remote_path in files_to_upload:
            sftp.put(local_path, remote_path)

        # Setup and run training
        # commands = [
        #     'sudo apt-get update -y',
        #     'sudo apt-get clean',
        #     'sudo rm -rf /var/lib/apt/lists/*',
        #     'sudo rm -rf /root/.cache/pip/*',
        #     'df -h',
        #     'mkdir -p /dev/shm/venv',  # Create virtual environment in tmpfs
        #     'python3 -m venv /dev/shm/venv',
        #     'source /dev/shm/venv/bin/activate',
        #     'pip install --no-cache-dir --target=/dev/shm/packages -r requirements.txt',  # Install packages to tmpfs
        #     'df -h',
        #     'export PYTHONPATH=/dev/shm/packages:$PYTHONPATH',  # Add packages to Python path
        #     'set -a && source .env && set +a',
        #     'python3 train_model.py'
        # ]
        commands = [
            'sudo apt-get update -y',
            # 'sudo apt-get install -y python3-pip',
            'sudo apt-get clean',  # Clean apt cache
            'sudo rm -rf /var/lib/apt/lists/*',  # Remove apt list files
            'sudo rm -rf /root/.cache/pip/*',  # Clear pip cache
            'df -h',  # Check available space
            ## 'sudo apt-get install -y python3-pip python3-venv',
            ## 'python3 -m venv .venv',
            ## 'source .venv/bin/activate',
            'pip install -r requirements.txt',
            # 'pip install --no-cache-dir -r requirements.txt',  # Avoid using pip cache
            'set -a && source .env && set +a',
            'python3 train_model.py'
        ]

        print("Setting up environment and starting training...")
        for cmd in commands:
            print(f"Running: {cmd}")
            stdin, stdout, stderr = ssh.exec_command(cmd)
            print(stdout.read().decode())
            print(stderr.read().decode())

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        ssh.close()

if __name__ == "__main__":
    deploy_training()