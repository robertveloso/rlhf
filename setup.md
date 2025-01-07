# Setup Guide for Llamabot Development Environment

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

## Setting up Virtual Environment

1. First, check your Python version:

   ```bash
   python3 --version
   ```

2. If you need to install Python, you can use the following command:

   ```bash
   sudo apt-get update
   sudo apt install python3 python3-pip
   python3 -m pip install --user virtualenv
   sudo apt-get install python3-venv
   ```

3. Create a new virtual environment:

   ```bash
   python3 -m venv .venv
   ```

4. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

   **On Windows:**

   ```bash
   source .venv\Scripts\activate
   ```

5. Install the required packages:

   ```bash
   python -m pip install --upgrade pip
   sudo apt-get install libpq-dev python3-dev
   pip install -r requirements.txt
   ```

6. Create and configure environment variables:
   ```bash
   touch .env
   ```
   Add the following to your `.env` file:
   ```
   OPENAI_API_KEY=your_openai_key
   PGPASSWORD=your_postgres_password
   SLACK_BOT_TOKEN=your_slack_bot_token
   SLACK_SIGNING_SECRET=your_slack_signing_secret
   ```

## Verifying Installation

1. Test the environment:

   ```bash
   python -c "import torch; print(torch.__version__)"
   python -c "import transformers; print(transformers.__version__)"
   ```

2. Start PostgreSQL (if using Docker):

   ```bash
   python start_postgres_docker.py
   ```

3. Run initial model training:
   ```bash
   python llamabot/train_initial_model.py
   ```

## Common Issues and Solutions

### GPU Support

If you have a CUDA-capable GPU, verify CUDA installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If not, you can install PyTorch without CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install --upgrade --force-reinstall -r requirements.txt
```
