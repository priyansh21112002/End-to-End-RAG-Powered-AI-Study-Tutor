"""
Inference module with 4-bit quantization
Optimized for 8GB VRAM with memory management
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any, Union, Iterator
import torch
from transformers import pipeline, TextStreamer
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from .model_loader import load_base_model, load_lora_model, get_model_info
from ..config import settings

logger = logging.getLogger(__name__)


class TutorLLM(LLM):
    """
    LLM wrapper optimized for tutoring tasks
    Handles inference with memory management for 8GB VRAM
    Compatible with LangChain Runnable interface
    """
    
    model_name: str = ""
    use_lora: bool = False
    device: str = "cuda"
    model: Any = None
    tokenizer: Any = None
    gen_config: Dict[str, Any] = {}
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_lora: bool = False,
        adapter_path: Optional[str] = None,
        use_quantization: bool = True,
        **kwargs
    ):
        """
        Initialize Tutor LLM
        
        Args:
            model_name: Model name/path
            use_lora: Whether to load LoRA adapters
            adapter_path: Path to LoRA adapters
            use_quantization: Use 4-bit quantization
        """
        super().__init__(**kwargs)
        self.model_name = model_name or settings.base_model_name
        self.use_lora = use_lora
        self.device = settings.get_device()
        
        # Load generation config
        config_path = settings.project_root / "src" / "config" / "model_config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        self.gen_config = config["inference"]
        
        # Load model and tokenizer
        logger.info("Initializing Tutor LLM...")
        if use_lora and adapter_path:
            self.model, self.tokenizer = load_lora_model(
                base_model_name=self.model_name,
                adapter_path=adapter_path
            )
        else:
            self.model, self.tokenizer = load_base_model(
                model_name=self.model_name,
                use_quantization=use_quantization
            )
        
        self.model.eval()  # Set to evaluation mode
        
        # Print model info
        logger.info("Model loaded successfully")
        self._print_memory_usage()
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM for LangChain"""
        return "tutor_llm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        LangChain-required method for running the LLM
        
        Args:
            prompt: Input prompt
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        return self.generate(prompt, stop=stop, **kwargs)
    
    def _print_memory_usage(self):
        """Print current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"VRAM - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate response from prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            stream: Whether to stream output (for terminal use)
            stop: Stop sequences (will be converted to stop token IDs)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.gen_config["max_new_tokens"]
        temperature = temperature or self.gen_config["temperature"]
        top_p = top_p or self.gen_config["top_p"]
        do_sample = do_sample if do_sample is not None else self.gen_config["do_sample"]
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": self.gen_config.get("top_k", 50),
            "do_sample": do_sample,
            "repetition_penalty": self.gen_config.get("repetition_penalty", 1.1),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        # Handle stop sequences
        if stop:
            # Convert stop strings to token IDs
            stop_token_ids = []
            for stop_seq in stop:
                tokens = self.tokenizer.encode(stop_seq, add_special_tokens=False)
                stop_token_ids.extend(tokens)
            if stop_token_ids:
                gen_kwargs["eos_token_id"] = list(set(stop_token_ids + [self.tokenizer.eos_token_id]))
        
        if stream:
            gen_kwargs["streamer"] = TextStreamer(self.tokenizer, skip_prompt=True)
        
        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Post-process: remove stop sequences from output if present
        if stop:
            for stop_seq in stop:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
        
        # Clear cache to free VRAM
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return generated_text.strip()
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        batch_size: int = 1
    ) -> List[str]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size (keep at 1 for 8GB VRAM)
        
        Returns:
            List of generated texts
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            for prompt in batch:
                result = self.generate(prompt, max_new_tokens=max_new_tokens)
                results.append(result)
        
        return results
    
    def _format_messages(self, messages: list) -> str:
        """Format list of messages into a single prompt"""
        formatted_parts = []
        
        for msg in messages:
            if hasattr(msg, 'content'):
                role = msg.__class__.__name__.replace('Message', '').replace('AI', 'Assistant')
                formatted_parts.append(f"{role}: {msg.content}")
            else:
                formatted_parts.append(str(msg))
        
        return "\n\n".join(formatted_parts)
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return get_model_info(self.model, self.tokenizer)
    
    def clear_memory(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
            self._print_memory_usage()


def create_inference_pipeline(
    model_name: Optional[str] = None,
    use_lora: bool = False,
    adapter_path: Optional[str] = None
) -> TutorLLM:
    """
    Create inference pipeline
    
    Args:
        model_name: Model name/path
        use_lora: Whether to use LoRA adapters
        adapter_path: Path to LoRA adapters
    
    Returns:
        TutorLLM instance
    """
    return TutorLLM(
        model_name=model_name,
        use_lora=use_lora,
        adapter_path=adapter_path
    )


if __name__ == "__main__":
    # Test inference
    logging.basicConfig(level=logging.INFO)
    
    print("Tutor LLM Inference Module")
    print("\nUsage:")
    print("  llm = TutorLLM()")
    print("  response = llm.generate('Explain gradient descent')")
    print("\nMemory-optimized for 8GB VRAM:")
    print("  - 4-bit quantization")
    print("  - Automatic cache clearing")
    print("  - Batch size 1 recommended")
    
    # Uncomment to test:
    # llm = TutorLLM()
    # print("\nModel Info:")
    # print(llm.get_info())
