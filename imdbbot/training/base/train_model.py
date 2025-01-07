from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from huggingface_hub import login
import os

def train():
    # Login to Hugging Face
    login(token=os.environ.get('HUGGINGFACE_TOKEN'))

    # Initialize model
    print("Initializing base BERT model...")
    # For portuguese: neuralmind/bert-base-portuguese-cased
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load IMDB dataset
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Training arguments with Hub push configuration
    training_args = TrainingArguments(
        output_dir="./models/movie-sentiment",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        # evaluation_strategy="steps", /home/user/.local/lib/python3.10/site-packages/transformers/training_args.py:1575: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        learning_rate=2e-5,
        push_to_hub=True,
        hub_model_id="robertveloso/movie-sentiment",  # Replace with your username/model-name
        hub_strategy="every_save"  # Pushes to hub at every save
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"]
    )

    # Train model
    print("Starting training...")
    trainer.train()

    # Save and push final model
    print("Saving and pushing model to Hugging Face Hub...")
    trainer.push_to_hub(
        commit_message="Final trained model",
        blocking=True  # Wait until the upload is complete
    )

    # Also push the tokenizer
    tokenizer.push_to_hub(
        "robertveloso/movie-sentiment",  # Replace with your username/model-name
        commit_message="Add tokenizer"
    )

    print("Model successfully pushed to Hugging Face Hub!")

if __name__ == "__main__":
    train()
