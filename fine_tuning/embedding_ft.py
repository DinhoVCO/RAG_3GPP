import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import argparse
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from data.embedding_train_data import load_and_prepare_datasets
from helpers.ir_evaluator import create_evaluator_information_retrieval




# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Displays in console
        logging.FileHandler("training.log")  # Saves to file
    ]
)


def parse_arguments():
    logging.info("Parsing command-line arguments...")
    parser = argparse.ArgumentParser(description="Embedding model training")
    parser.add_argument('--model_name', type=str, default="BAAI/bge-small-en-v1.5", help="Base model name")
    parser.add_argument('--new_model_name', type=str, default="BAAI/bge-small-en-v1.5", help="Base new model name")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--output_dir', type=str, default="/models/embedding/", help="Output directory for the trained model")
    parser.add_argument('--seed', type=int, default=42, help="Seed for reproducibility")
    args = parser.parse_args()
    logging.info("Arguments parsed successfully.")
    return args



#model_name = "BAAI/bge-small-en-v1.5"
#epochs = 5
#batch_size = 128
#my_model_name = "bge-small-telecom"+f"_{str(epochs)}e_{str(batch_size)}bs"
#path_output = "/models/embedding/"


def initialize_model(model_name):
    logging.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    loss = MultipleNegativesRankingLoss(model)
    logging.info("Model and loss function initialized.")
    return model, loss

def configure_training(my_model_name, epochs, batch_size, output_dir):
    logging.info("Configuring training arguments...")
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir+f"checkpoints/{my_model_name}",
        # Optional training parameters:
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        learning_rate= 5e-5,#2e-05, #3e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine_with_restarts",
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=15,
        save_strategy="steps",
        save_steps=15,
        save_total_limit=1,
        logging_steps=15,
        run_name=my_model_name,  # Will be used in W&B if wandb is installed
        load_best_model_at_end=True,
        metric_for_best_model="eval_telecom-ir-eval_cosine_accuracy@1",
        report_to= "wandb" #"none" ,
    )
    logging.info("Training arguments configured.")
    return args


def train_model(model, args_training, train_dataset, val_dataset, loss, evaluator):
    logging.info("Starting training process...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args_training,
        train_dataset=train_dataset.select_columns(["anchor", "positive"]),
        eval_dataset=val_dataset.select_columns(["anchor", "positive"]),
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    logging.info("Training completed.")
    return trainer

def save_model(trainer, output_dir, model_name):
    logging.info("Saving the trained model...")
    trainer.model.save_pretrained(os.path.join(output_dir, model_name))
    logging.info("Model saved successfully.")


# Main function
def main():
    logging.info("Starting main process...")
    args = parse_arguments()
    train_dataset, val_dataset, test_dataset = load_and_prepare_datasets(args.seed)
    evaluator = create_evaluator_information_retrieval(val_dataset)
    my_model_name = f"{str(args.new_model_name)}_{str(args.epochs)}e_{str(args.batch_size)}bs"
    model, loss = initialize_model(args.model_name)
    args_training = configure_training(my_model_name, args.epochs, args.batch_size, args.output_dir)
    trainer = train_model(model, args_training, train_dataset, val_dataset, loss, evaluator)
    save_model(trainer, args.output_dir, my_model_name)
    logging.info("Process completed successfully.")

if __name__ == "__main__":
    main()