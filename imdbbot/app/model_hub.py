from huggingface_hub import HfApi, snapshot_download
import os

class ModelHub:
    def __init__(self, model_id="robertveloso/movie-sentiment"):
        self.api = HfApi()
        self.model_id = model_id
        self.local_dir = "./models"
        os.makedirs(self.local_dir, exist_ok=True)

    def push_model(self, model_path, commit_message="Update model"):
        """Push trained model to Hugging Face Hub"""
        self.api.upload_folder(
            folder_path=model_path,
            repo_id=self.model_id,
            commit_message=commit_message
        )

    def get_latest_model(self):
        """Download latest model from Hub"""
        return snapshot_download(
            repo_id=self.model_id,
            cache_dir=self.local_dir
        )