import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Any
from dbos import DBOS
from .model_hub import ModelHub

class SentimentModel:
    def __init__(self, model_id="robertveloso/movie-sentiment"):
        self.hub = ModelHub(model_id=model_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = self.hub.get_latest_model()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)

    def predict(self, text: str) -> int:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1)

        return prediction.item()

    @DBOS.workflow()
    def process_feedback(self, feedback: Dict[str, Any]):
        if feedback.get("user_feedback") is None:
            return

        text = feedback["text"]
        rating = feedback["rating"]

        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get current prediction
        self.model.eval()
        with torch.no_grad():
            initial_outputs = self.model(**inputs)

        # Train step with user feedback
        self.model.train()
        outputs = self.model(**inputs)

        # Calculate loss based on user feedback
        loss = self._calculate_rlhf_loss(outputs, initial_outputs, rating)

        # Update model
        loss.backward()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        optimizer.step()
        optimizer.zero_grad()

        self.model.eval()

    def _calculate_rlhf_loss(self, outputs, initial_outputs, rating):
        # Implementation based on existing RLHF logic
        # Reference the original implementation from:
        startLine: 60
        endLine: 67

        prob_ratio = torch.softmax(outputs.logits, dim=1) / torch.softmax(initial_outputs.logits, dim=1)
        clipped_ratio = torch.clamp(prob_ratio, 0.8, 1.2)
        scaled_reward = (rating - 2.5) / 2.5

        return -torch.min(prob_ratio * scaled_reward, clipped_ratio * scaled_reward).mean()
