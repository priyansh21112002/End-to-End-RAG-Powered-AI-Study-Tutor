"""
Model loader with 4-bit quantization support
Handles loading base models and LoRA adapters optimized for 8GB VRAM
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
from ..config import settings

logger = logging.getLogger(__name__)


def get_quantization_config() -> BitsAndBytesConfig:
    """
    Get 4-bit quantization configuration optimized for 8GB VRAM
    
    Returns:
        BitsAndBytesConfig for 4-bit quantization
    """
    # Load config from JSON
    config_path = settings.project_root / "src" / "config" / "model_config.json"
    
    with open(config_path) as f:
        config = json.load(f)
    
    quant_config = config["quantization"]
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config["load_in_4bit"],
        bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_config["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"]
    )
    
    logger.info("Created 4-bit quantization config")
    return bnb_config


def load_base_model(
    model_name: Optional[str] = None,
    use_quantization: bool = True,
    device_map: str = "auto",
    trust_remote_code: bool = False
) -> tuple:
    """
    Load base model with optional quantization
    
    Args:
        model_name: Model name/path (defaults to settings)
        use_quantization: Whether to use 4-bit quantization
        device_map: Device mapping strategy
        trust_remote_code: Trust remote code for some models
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = model_name or settings.base_model_name
    
    logger.info(f"Loading base model: {model_name}")
    logger.info(f"Quantization: {use_quantization}")
    logger.info(f"Device: {settings.get_device()}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(settings.model_cache_dir),
        trust_remote_code=trust_remote_code,
        token=settings.hf_token
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Model loading kwargs
    model_kwargs = {
        "cache_dir": str(settings.model_cache_dir),
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "token": settings.hf_token
    }
    
    if use_quantization:
        model_kwargs["quantization_config"] = get_quantization_config()
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["low_cpu_mem_usage"] = True
        model_kwargs["max_memory"] = {0: "7GB", "cpu": "8GB"}
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Print model info
    if torch.cuda.is_available():
        logger.info(f"Model loaded on GPU")
        logger.info(f"VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return model, tokenizer


def load_lora_model(
    base_model_name: Optional[str] = None,
    adapter_path: Optional[str] = None,
    merge_weights: bool = False
) -> tuple:
    """
    Load model with LoRA adapters
    
    Args:
        base_model_name: Base model name
        adapter_path: Path to LoRA adapter weights
        merge_weights: Whether to merge adapters into base model
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if adapter_path is None:
        # Look for latest checkpoint
        checkpoint_dir = settings.lora_adapter_dir
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"No LoRA adapters found at {checkpoint_dir}")
        
        # Find latest adapter
        adapters = sorted(checkpoint_dir.glob("checkpoint-*"))
        if not adapters:
            raise FileNotFoundError(f"No checkpoint folders found in {checkpoint_dir}")
        
        adapter_path = str(adapters[-1])
        logger.info(f"Using latest adapter: {adapter_path}")
    
    base_model_name = base_model_name or settings.base_model_name
    
    logger.info(f"Loading LoRA model from {adapter_path}")
    
    # Load base model with quantization
    base_model, tokenizer = load_base_model(
        model_name=base_model_name,
        use_quantization=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path
    )
    
    if merge_weights:
        logger.info("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
    
    logger.info("LoRA model loaded successfully")
    
    if torch.cuda.is_available():
        logger.info(f"VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return model, tokenizer


def get_model_info(model, tokenizer) -> Dict[str, Any]:
    """
    Get model information and statistics
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
    
    Returns:
        Dictionary with model info
    """
    info = {
        "model_type": model.config.model_type,
        "vocab_size": tokenizer.vocab_size,
        "max_position_embeddings": getattr(model.config, "max_position_embeddings", "N/A"),
        "is_quantized": hasattr(model, "quantization_config"),
        "device": str(model.device)
    }
    
    if torch.cuda.is_available():
        info["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
        info["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
        info["gpu_name"] = torch.cuda.get_device_name(0)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info["total_parameters"] = f"{total_params:,}"
    info["trainable_parameters"] = f"{trainable_params:,}"
    info["trainable_percentage"] = f"{100 * trainable_params / total_params:.2f}%"
    
    return info


def print_model_info(model, tokenizer):
    """Print model information"""
    info = get_model_info(model, tokenizer)
    
    print("\n=== Model Information ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    print("=" * 30 + "\n")


if __name__ == "__main__":
    # Test model loading
    logging.basicConfig(level=logging.INFO)
    
    print("Model Loader Utility")
    print("\nFunctions:")
    print("- load_base_model(): Load base model with 4-bit quantization")
    print("- load_lora_model(): Load model with LoRA adapters")
    print("- get_model_info(): Get model statistics")
    
    # Uncomment to test:
    # model, tokenizer = load_base_model()
    # print_model_info(model, tokenizer)
