import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from RAG.format_prompt import get_full_promt

def load_and_prepare_dataset(dataset_name):
    train_dataset = load_dataset(dataset_name, split='train')
    dataset = train_dataset.map(lambda row: {'text': get_full_promt(row, True)})
    return dataset

def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        revision="refs/pr/23"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def configure_lora():
    return LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["Wqkv", "fc1", "fc2"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

def configure_training_arguments(output_dir, batch_size, num_epochs):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        fp16=True,
        bf16=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        learning_rate=2e-4,
        weight_decay=0.001,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        save_steps=0,
        logging_steps=25,
        report_to="none"
    )

def train_model(batch_size, model_name, new_model_name, save_path, num_epochs,train_dataset_name):
    dataset = load_and_prepare_dataset('train_dataset_name', split='train')
    model, tokenizer = load_model_and_tokenizer(model_name)
    peft_config = configure_lora()

    collator = DataCollatorForCompletionOnlyLM("Output:", tokenizer=tokenizer)

    training_arguments = configure_training_arguments(
        os.path.join(save_path, new_model_name), batch_size, num_epochs
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        peft_config=peft_config,
        data_collator=collator,
    )
    
    trainer.train()
    trainer.model.save_pretrained(os.path.join(save_path, new_model_name))

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-2 model with LoRA.")
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--model_name', type=str, default='microsoft/phi-2', help='Pretrained model name')
    parser.add_argument('--new_model_name', type=str, default='phi-2-3GPP-RAG-ft' , help='Name for the fine-tuned model')
    parser.add_argument('--train_dataset_name', type=str, default='dinho1597/3GPP_QA_RAG', help='Name for the dataset for training')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the fine-tuned model')
    parser.add_argument('--num_epochs', type=int, default=4, help='Number of training epochs')

    args = parser.parse_args()

    train_model(
        batch_size=args.batch_size,
        model_name=args.model_name,
        new_model_name=args.new_model_name,
        save_path=args.save_path,
        num_epochs=args.num_epochs,
        train_dataset_name=args.train_dataset_name
    )

if __name__ == "__main__":
    main()
