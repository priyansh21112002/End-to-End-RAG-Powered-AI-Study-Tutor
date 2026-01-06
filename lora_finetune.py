"""
QLoRA Fine-tuning Script
Optimized for 8GB VRAM using 4-bit quantization and memory-efficient training
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTTrainer
from ..config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_config() -> Dict[str, Any]:
    """Load training configuration from JSON"""
    config_path = settings.project_root / "src" / "config" / "model_config.json"
    
    with open(config_path) as f:
        config = json.load(f)
    
    return config


def create_quantization_config(config: Dict[str, Any]) -> BitsAndBytesConfig:
    """
    Create 4-bit quantization config for training
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        BitsAndBytesConfig
    """
    quant_config = config["quantization"]
    
    return BitsAndBytesConfig(
        load_in_4bit=quant_config["load_in_4bit"],
        bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_config["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"]
    )


def create_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """
    Create LoRA configuration
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        LoraConfig
    """
    lora_config = config["lora"]
    
    return LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        task_type=lora_config["task_type"]
    )


def create_training_arguments(
    config: Dict[str, Any],
    output_dir: Optional[str] = None
) -> TrainingArguments:
    """
    Create training arguments optimized for 8GB VRAM
    
    Args:
        config: Model configuration dictionary
        output_dir: Output directory for checkpoints
    
    Returns:
        TrainingArguments
    """
    training_config = config["training"]
    output_dir = output_dir or str(settings.checkpoint_dir)
    
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        max_steps=training_config["max_steps"],
        learning_rate=training_config["learning_rate"],
        fp16=training_config["fp16"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        warmup_steps=training_config["warmup_steps"],
        optim=training_config["optim"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        max_grad_norm=training_config["max_grad_norm"],
        save_total_limit=3,  # Keep only 3 checkpoints
        logging_dir=f"{output_dir}/logs",
        report_to="none",  # Disable wandb/tensorboard for now
        remove_unused_columns=False,
        push_to_hub=False
    )


def format_instruction_prompt(example: Dict[str, str]) -> str:
    """
    Format training example into instruction prompt
    
    Args:
        example: Dictionary with 'instruction', 'input', 'output' keys
    
    Returns:
        Formatted prompt string
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")
    
    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
{output_text}"""
    
    return prompt


def load_training_data(data_path: str) -> Dataset:
    """
    Load training data from JSONL file
    
    Args:
        data_path: Path to training data file
    
    Returns:
        HuggingFace Dataset
    """
    logger.info(f"Loading training data from {data_path}")
    
    if data_path.endswith('.jsonl') or data_path.endswith('.json'):
        dataset = load_dataset('json', data_files=data_path, split='train')
    else:
        raise ValueError(f"Unsupported data format: {data_path}")
    
    logger.info(f"Loaded {len(dataset)} training examples")
    return dataset


def prepare_model_for_training(
    model_name: Optional[str] = None,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    lora_config: Optional[LoraConfig] = None
) -> tuple:
    """
    Prepare model and tokenizer for QLoRA training
    
    Args:
        model_name: Model name/path
        quantization_config: Quantization configuration
        lora_config: LoRA configuration
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = model_name or settings.base_model_name
    
    logger.info(f"Loading base model: {model_name}")
    logger.info(f"Device: {settings.get_device()}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(settings.model_cache_dir),
        trust_remote_code=False
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = "right"  # Required for some models
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        cache_dir=str(settings.model_cache_dir),
        trust_remote_code=False,
        torch_dtype=torch.float16
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"VRAM allocated: {allocated:.2f} GB")
    
    return model, tokenizer


def train_model(
    model,
    tokenizer,
    dataset: Dataset,
    training_args: TrainingArguments,
    max_seq_length: int = 2048
) -> SFTTrainer:
    """
    Train model using SFTTrainer
    
    Args:
        model: Prepared model with LoRA
        tokenizer: Tokenizer
        dataset: Training dataset
        training_args: Training arguments
        max_seq_length: Maximum sequence length
    
    Returns:
        Trained SFTTrainer
    """
    logger.info("Starting training...")
    logger.info(f"Max sequence length: {max_seq_length}")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=max_seq_length,
        formatting_func=format_instruction_prompt,
        packing=False  # Disable packing to save memory
    )
    
    # Train
    trainer.train()
    
    logger.info("Training completed!")
    
    return trainer


def save_model(
    trainer: SFTTrainer,
    output_dir: Optional[str] = None
):
    """
    Save trained model and adapters
    
    Args:
        trainer: Trained SFTTrainer
        output_dir: Output directory for final model
    """
    output_dir = output_dir or str(settings.lora_adapter_dir / "final")
    
    logger.info(f"Saving model to {output_dir}")
    
    # Save LoRA adapters
    trainer.model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)
    
    logger.info("Model saved successfully!")


def run_training(
    data_path: str,
    model_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_steps: Optional[int] = None
):
    """
    Main training function
    
    Args:
        data_path: Path to training data (JSONL)
        model_name: Model name/path
        output_dir: Output directory for checkpoints
        max_steps: Override max training steps
    """
    logger.info("=== QLoRA Fine-tuning ===")
    logger.info(f"Data: {data_path}")
    logger.info(f"Model: {model_name or settings.base_model_name}")
    
    # Load configs
    config = load_training_config()
    
    # Override max_steps if provided
    if max_steps:
        config["training"]["max_steps"] = max_steps
    
    # Create configurations
    quantization_config = create_quantization_config(config)
    lora_config = create_lora_config(config)
    training_args = create_training_arguments(config, output_dir)
    
    # Load data
    dataset = load_training_data(data_path)
    
    # Prepare model
    model, tokenizer = prepare_model_for_training(
        model_name=model_name,
        quantization_config=quantization_config,
        lora_config=lora_config
    )
    
    # Train
    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_args=training_args,
        max_seq_length=config["memory_optimization"]["max_sequence_length"]
    )
    
    # Save final model
    save_model(trainer, output_dir)
    
    logger.info("=== Training Complete ===")
    
    if torch.cuda.is_available():
        final_vram = torch.cuda.memory_allocated() / 1024**3
        max_vram = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"Final VRAM: {final_vram:.2f} GB")
        logger.info(f"Peak VRAM: {max_vram:.2f} GB")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune LLM with QLoRA")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data (JSONL format)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps"
    )
    
    args = parser.parse_args()
    
    run_training(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        max_steps=args.max_steps
    )
