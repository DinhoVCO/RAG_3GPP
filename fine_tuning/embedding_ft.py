import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from data.embedding_train_data import load_and_prepare_datasets
from helpers.ir_evaluator import evaluate_information_retrieval

seed=42


print('Loading and preparing datasets...')
#Dataset
train_dataset, val_dataset, test_dataset = load_and_prepare_datasets(seed)

print('Datasets loaded and prepared.')

print('creating evaluator...')
# Evaluator
evaluator = evaluate_information_retrieval(test_dataset)
 

print('TRAINING STARTED...')
#TRAINING

model_name = "BAAI/bge-small-en-v1.5"
epochs = 5
batch_size = 128
my_model_name = "bge-small-telecom"+f"_{str(epochs)}e_{str(batch_size)}bs"
path_output = "/models/embedding/"

print('loaded model:', model_name)
model = SentenceTransformer(model_name)
loss = MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=path_output+f"checkpoints/{my_model_name}",
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
    run_name=my_model_name,  # Will be used in W&B if `wandb` is installed
    load_best_model_at_end=True,
    metric_for_best_model="eval_telecom-ir-eval_cosine_accuracy@1",
    report_to= "wandb" #"none" ,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset.select_columns(["anchor", "positive"]),
    eval_dataset=val_dataset.select_columns(["anchor", "positive"]),
    loss=loss,
    evaluator=evaluator,
)

print('Training...')
trainer.train()
print('Training finished.')

print('Saving model...')
trainer.model.save_pretrained(path_output+my_model_name)
print('Model saved.')