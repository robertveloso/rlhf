from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SFTTrainer:
    def __init__(self, model_id="robertveloso/movie-sentiment"):
        self.model_id = model_id
        # Load tokenizer from the pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Ensure we're logged in to Hugging Face
        if "HUGGINGFACE_TOKEN" not in os.environ:
            raise ValueError("HUGGINGFACE_TOKEN environment variable is required")
        login(token=os.environ["HUGGINGFACE_TOKEN"])

        logger.info(f"Initializing SFT trainer with model ID: {model_id}")

    def prepare_datasets(self):
        """Load and prepare Reddit datasets for fine-tuning"""
        logger.info("Loading Reddit dataset...")

        # Load Reddit data from CSV files
        try:
            reddit = load_dataset("csv", data_files="reddit_data_*.csv")
            logger.info("Successfully loaded Reddit data")
        except Exception as e:
            logger.error(f"Could not load Reddit data: {e}")
            raise ValueError("Reddit data is required for fine-tuning")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )

        logger.info("Tokenizing dataset...")
        reddit_tokenized = reddit.map(tokenize_function, batched=True)

        return reddit_tokenized["train"].train_test_split(test_size=0.1)

    def train(self):
        """Run the SFT training process"""
        logger.info("Starting SFT training process...")

        # Load the pre-trained model
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)

        # Prepare datasets
        datasets = self.prepare_datasets()

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir="./checkpoints/sft",
            push_to_hub=True,
            hub_model_id=self.model_id,
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=500,
            learning_rate=2e-5,  # Using a lower learning rate for fine-tuning
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir="./logs/sft",
            logging_steps=100
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"]
        )

        # Run training
        logger.info("Starting fine-tuning...")
        trainer.train()

        # Push to hub
        logger.info("Fine-tuning complete. Pushing to Hugging Face Hub...")
        trainer.push_to_hub("Updated model with Reddit data fine-tuning")
        logger.info(f"Model successfully pushed to: {self.model_id}")

def main():
    """Main entry point for SFT training"""
    try:
        trainer = SFTTrainer()
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()