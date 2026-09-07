#!/usr/bin/env python3
"""
SFT Training Script for Llama Models
Supports LoRA fine-tuning with various configurations.
"""

import os
import json
import torch
import argparse
from typing import Dict, Any
from pathlib import Path

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset


class SFTTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        print(f"Loading model from {self.config['model_name_or_path']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name_or_path'],
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name_or_path'],
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto" if self.config.get('use_device_map', True) else None
        )
        
        # Prepare model for training
        if self.config.get('use_4bit', False):
            self.model = prepare_model_for_kbit_training(self.model)
            
    def setup_lora(self):
        """Setup LoRA configuration."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get('lora_r', 16),
            lora_alpha=self.config.get('lora_alpha', 32),
            lora_dropout=self.config.get('lora_dropout', 0.1),
            target_modules=self.config.get('target_modules', [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def load_training_data(self):
        """Load and prepare training data."""
        print(f"Loading training data from {self.config['train_data_path']}")
        
        with open(self.config['train_data_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Convert to instruction format
        formatted_data = []
        for item in data:
            instruction = item['instruction']
            response = item['response']
            
            # Format for Llama instruction following
            prompt = f"<|im_start|>system\nYou are a helpful coding tutor assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            
            formatted_data.append({
                "text": prompt,
                "instruction": instruction,
                "response": response
            })
            
        self.train_dataset = Dataset.from_list(formatted_data)
        print(f"Loaded {len(self.train_dataset)} training samples")
        
    def tokenize_function(self, examples):
        """Tokenize the training examples."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=self.config.get('max_length', 2048),
            return_tensors="pt"
        )
        
    def train(self):
        """Run the training process."""
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Setup LoRA
        self.setup_lora()
        
        # Load training data
        self.load_training_data()
        
        # Tokenize dataset
        tokenized_dataset = self.train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config.get('num_epochs', 3),
            per_device_train_batch_size=self.config.get('batch_size', 4),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 4),
            learning_rate=self.config.get('learning_rate', 2e-4),
            warmup_steps=self.config.get('warmup_steps', 100),
            logging_steps=self.config.get('logging_steps', 10),
            save_steps=self.config.get('save_steps', 500),
            evaluation_strategy="no",
            save_strategy="steps",
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if self.config.get('use_wandb', False) else None,
        )
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        print("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config['output_dir'])
        print(f"Training completed. Model saved to {self.config['output_dir']}")


def main():
    parser = argparse.ArgumentParser(description="SFT Training for Llama Models")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to training configuration JSON file")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize and run trainer
    trainer = SFTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 