"""
Performance monitoring utilities
Real-time tracking of system metrics during inference
"""

import time
import torch
import psutil
from typing import Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class PerformanceStats:
    """Container for performance statistics"""
    inference_time: float
    retrieval_time: float
    total_time: float
    vram_used_gb: float
    vram_peak_gb: float
    cpu_percent: float
    tokens_generated: int
    tokens_per_second: float


class PerformanceMonitor:
    """
    Monitor and track performance metrics in real-time
    """
    
    def __init__(self):
        self.has_cuda = torch.cuda.is_available()
        self.reset()
    
    def reset(self):
        """Reset all counters"""
        self.total_queries = 0
        self.total_inference_time = 0.0
        self.total_retrieval_time = 0.0
        self.total_tokens = 0
        self.vram_samples = []
        
        if self.has_cuda:
            torch.cuda.reset_peak_memory_stats()
    
    @contextmanager
    def track_inference(self):
        """Context manager to track inference metrics"""
        start_time = time.time()
        
        if self.has_cuda:
            torch.cuda.synchronize()
            start_vram = torch.cuda.memory_allocated()
        
        try:
            yield
        finally:
            if self.has_cuda:
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            self.total_inference_time += elapsed
            self.total_queries += 1
            
            if self.has_cuda:
                current_vram = torch.cuda.memory_allocated()
                self.vram_samples.append(current_vram / 1024**3)  # Convert to GB
    
    @contextmanager
    def track_retrieval(self):
        """Context manager to track retrieval metrics"""
        start_time = time.time()
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.total_retrieval_time += elapsed
    
    def get_current_stats(self, num_tokens: int = 0) -> PerformanceStats:
        """Get current performance statistics"""
        inference_time = (
            self.total_inference_time / self.total_queries 
            if self.total_queries > 0 else 0.0
        )
        
        retrieval_time = (
            self.total_retrieval_time / self.total_queries 
            if self.total_queries > 0 else 0.0
        )
        
        total_time = inference_time + retrieval_time
        
        if self.has_cuda:
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_peak = torch.cuda.max_memory_allocated() / 1024**3
        else:
            vram_used = 0.0
            vram_peak = 0.0
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        tokens_per_sec = (
            num_tokens / inference_time 
            if inference_time > 0 else 0.0
        )
        
        return PerformanceStats(
            inference_time=inference_time,
            retrieval_time=retrieval_time,
            total_time=total_time,
            vram_used_gb=vram_used,
            vram_peak_gb=vram_peak,
            cpu_percent=cpu_percent,
            tokens_generated=num_tokens,
            tokens_per_second=tokens_per_sec
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        avg_inference = (
            self.total_inference_time / self.total_queries 
            if self.total_queries > 0 else 0.0
        )
        
        avg_retrieval = (
            self.total_retrieval_time / self.total_queries 
            if self.total_queries > 0 else 0.0
        )
        
        avg_vram = (
            sum(self.vram_samples) / len(self.vram_samples) 
            if self.vram_samples else 0.0
        )
        
        if self.has_cuda:
            peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        else:
            peak_vram = 0.0
        
        return {
            'total_queries': self.total_queries,
            'avg_inference_time': avg_inference,
            'avg_retrieval_time': avg_retrieval,
            'avg_total_time': avg_inference + avg_retrieval,
            'avg_vram_gb': avg_vram,
            'peak_vram_gb': peak_vram,
            'total_time': self.total_inference_time + self.total_retrieval_time
        }
    
    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()
        
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Total Queries:        {summary['total_queries']}")
        print(f"Avg Inference Time:   {summary['avg_inference_time']:.3f}s")
        print(f"Avg Retrieval Time:   {summary['avg_retrieval_time']:.3f}s")
        print(f"Avg Total Time:       {summary['avg_total_time']:.3f}s")
        
        if self.has_cuda:
            print(f"Avg VRAM Usage:       {summary['avg_vram_gb']:.2f}GB")
            print(f"Peak VRAM Usage:      {summary['peak_vram_gb']:.2f}GB")
        
        print("="*50 + "\n")
