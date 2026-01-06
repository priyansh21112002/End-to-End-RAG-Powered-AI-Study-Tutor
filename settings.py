"""
Configuration management for Study Tutor AI
Handles settings, model paths, and hyperparameters optimized for 8GB VRAM
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Model Configuration
    base_model_name: str = Field(
        default="meta-llama/Llama-3.2-3B-Instruct",
        description="Base LLM model (optimized for 8GB VRAM)"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for vector store"
    )
    
    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("./data"))
    raw_data_dir: Path = Field(default_factory=lambda: Path("./data/raw"))
    processed_data_dir: Path = Field(default_factory=lambda: Path("./data/processed"))
    vectorstore_dir: Path = Field(default_factory=lambda: Path("./data/vectorstore"))
    model_cache_dir: Path = Field(default_factory=lambda: Path("./models/cache"))
    checkpoint_dir: Path = Field(default_factory=lambda: Path("./models/checkpoints"))
    lora_adapter_dir: Path = Field(default_factory=lambda: Path("./models/lora_adapters"))
    
    # GPU Settings
    cuda_visible_devices: str = Field(default="0")
    use_flash_attention: bool = Field(default=False, description="Requires flash-attn package")
    
    # RAG Settings
    chunk_size: int = Field(default=800, description="Chunk size in tokens")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks")
    retrieval_k: int = Field(default=5, description="Number of documents to retrieve")
    
    # Training Settings
    max_steps: int = Field(default=500, description="Training steps")
    learning_rate: float = Field(default=2e-4)
    lora_r: int = Field(default=16, description="LoRA rank")
    lora_alpha: int = Field(default=32, description="LoRA alpha")
    
    # API Settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    
    # Optional
    hf_token: Optional[str] = Field(default=None, description="Hugging Face token for gated models")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.vectorstore_dir,
            self.model_cache_dir,
            self.checkpoint_dir,
            self.lora_adapter_dir
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_device(self) -> str:
        """Get compute device (cuda/cpu)"""
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def print_gpu_info(self):
        """Print GPU information"""
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("No GPU available, using CPU")


# Global settings instance
settings = Settings()


if __name__ == "__main__":
    # Test configuration
    print("=== Study Tutor AI Configuration ===")
    print(f"Base Model: {settings.base_model_name}")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"Project Root: {settings.project_root}")
    print(f"Vector Store: {settings.vectorstore_dir}")
    settings.print_gpu_info()
