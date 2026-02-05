"""
Sample Queue for Drifting Field Generative Models

This module implements efficient sample queues for storing and retrieving
positive samples (attractors) during training, as described in Appendix A.8.

Key features:
- FIFO queue per class for ImageNet (1000 classes + 1 unconditional)
- GPU-resident tensors to avoid CPU-GPU synchronization
- Efficient batch operations for enqueueing and sampling
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Union
from collections import deque


class GlobalSampleQueue:
    """
    Global sample queue for unconditional samples.
    
    This is a simple FIFO queue that stores samples from all classes,
    used for Classifier-Free Guidance training.
    
    Args:
        queue_size: Maximum number of samples to store
        latent_shape: Shape of latent samples (C, H, W) or (D,) for flat vectors
        device: Device to store tensors on
    """
    
    def __init__(
        self,
        queue_size: int = 1000,
        latent_shape: Tuple[int, ...] = (4, 32, 32),
        device: str = "cpu",
    ):
        self.queue_size = queue_size
        self.latent_shape = latent_shape
        self.device = device
        
        # Pre-allocate buffer for efficiency
        self.buffer = torch.zeros(queue_size, *latent_shape, device=device)
        self.ptr = 0  # Write pointer
        self.count = 0  # Number of valid samples
    
    def __len__(self) -> int:
        """Return number of valid samples in queue."""
        return self.count
    
    def add(self, samples: torch.Tensor) -> None:
        """
        Add samples to the queue.
        
        Args:
            samples: Samples to add of shape (B, *latent_shape)
        """
        samples = samples.to(self.device)
        batch_size = samples.shape[0]
        
        if batch_size >= self.queue_size:
            # If batch is larger than queue, just take the last queue_size samples
            self.buffer[:] = samples[-self.queue_size:]
            self.ptr = 0
            self.count = self.queue_size
        else:
            # Calculate how many samples we can add without wrapping
            remaining = self.queue_size - self.ptr
            
            if batch_size <= remaining:
                # No wrap needed
                self.buffer[self.ptr:self.ptr + batch_size] = samples
            else:
                # Need to wrap around
                self.buffer[self.ptr:] = samples[:remaining]
                self.buffer[:batch_size - remaining] = samples[remaining:]
            
            self.ptr = (self.ptr + batch_size) % self.queue_size
            self.count = min(self.count + batch_size, self.queue_size)
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Sample from the queue.
        
        Args:
            num_samples: Number of samples to retrieve
            
        Returns:
            Sampled tensors of shape (num_samples, *latent_shape)
        """
        if self.count == 0:
            # Return random samples if queue is empty
            return torch.randn(num_samples, *self.latent_shape, device=self.device)
        
        # Sample with replacement if needed
        indices = torch.randint(0, self.count, (num_samples,), device=self.device)
        return self.buffer[indices].clone()
    
    def is_ready(self, min_samples: int = 32) -> bool:
        """Check if queue has enough samples."""
        return self.count >= min_samples
    
    def clear(self) -> None:
        """Clear the queue."""
        self.ptr = 0
        self.count = 0


class SampleQueue:
    """
    Class-aware sample queue for conditional generative models.
    
    Maintains a separate FIFO queue for each class, plus a global
    unconditional queue for CFG training.
    
    This implements the sample queue described in Appendix A.8:
    - Each class has a queue of size 128 (configurable)
    - Global unconditional queue of size 1000 (configurable)
    
    Args:
        queue_size: Maximum samples per class
        num_classes: Number of classes (default: 1000 for ImageNet)
        latent_shape: Shape of latent samples (C, H, W)
        device: Device to store tensors on
        unconditional_queue_size: Size of the unconditional queue
    """
    
    def __init__(
        self,
        queue_size: int = 128,
        num_classes: int = 1000,
        latent_shape: Tuple[int, ...] = (4, 32, 32),
        device: str = "cpu",
        unconditional_queue_size: int = 1000,
    ):
        self.queue_size = queue_size
        self.num_classes = num_classes
        self.latent_shape = latent_shape
        self.device = device
        
        # Create per-class queues (using lists for simplicity)
        # +1 for unconditional class
        self.queues: List[deque] = [
            deque(maxlen=queue_size) for _ in range(num_classes + 1)
        ]
        
        # Unconditional queue (global, larger)
        self.unconditional_queue = GlobalSampleQueue(
            queue_size=unconditional_queue_size,
            latent_shape=latent_shape,
            device=device,
        )
    
    def add(self, samples: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Add samples to their respective class queues.
        
        Args:
            samples: Samples of shape (B, *latent_shape)
            labels: Class labels of shape (B,)
        """
        samples = samples.to(self.device)
        labels = labels.to(self.device)
        
        # Group samples by class for efficient batching
        labels_cpu = labels.cpu().numpy()
        for class_idx in range(len(self.queues)):
            mask = labels_cpu == class_idx
            if mask.any():
                class_samples = samples[mask]
                for sample in class_samples:
                    self.queues[class_idx].append(sample.clone())
    
    def add_unconditional(self, samples: torch.Tensor) -> None:
        """
        Add samples to the unconditional queue.
        
        Args:
            samples: Samples of shape (B, *latent_shape)
        """
        self.unconditional_queue.add(samples)
    
    def sample(
        self,
        labels: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample from queues based on class labels.
        
        For each label in the batch, retrieves samples from the
        corresponding class queue.
        
        Args:
            labels: Class labels of shape (B,)
            num_samples: Number of samples per class (currently supports 1)
            
        Returns:
            Sampled tensors of shape (B, *latent_shape)
        """
        import random
        
        batch_size = labels.shape[0]
        samples = torch.zeros(batch_size, *self.latent_shape, device=self.device)
        labels_cpu = labels.cpu().tolist()
        
        for i, class_idx in enumerate(labels_cpu):
            queue = self.queues[class_idx] if class_idx < len(self.queues) else None
            
            if queue is not None and len(queue) > 0:
                # Randomly sample from the class queue using random.randint
                idx = random.randint(0, len(queue) - 1)
                samples[i] = queue[idx]  # deque supports direct indexing
            else:
                # Fallback to random sample if queue is empty
                samples[i] = torch.randn(*self.latent_shape, device=self.device)
        
        return samples
    
    def sample_unconditional(self, num_samples: int) -> torch.Tensor:
        """
        Sample from the unconditional queue.
        
        Args:
            num_samples: Number of samples to retrieve
            
        Returns:
            Sampled tensors of shape (num_samples, *latent_shape)
        """
        return self.unconditional_queue.sample(num_samples)
    
    def is_ready(self, min_samples: int = 32) -> bool:
        """
        Check if queues have enough samples for training.
        
        Args:
            min_samples: Minimum total samples across all queues
            
        Returns:
            True if ready for training
        """
        total = sum(len(q) for q in self.queues)
        return total >= min_samples
    
    def stats(self) -> Dict[str, Union[int, float]]:
        """
        Get queue statistics.
        
        Returns:
            Dictionary with queue statistics
        """
        queue_sizes = [len(q) for q in self.queues]
        non_empty = sum(1 for s in queue_sizes if s > 0)
        
        return {
            "total_samples": sum(queue_sizes),
            "num_non_empty": non_empty,
            "max_queue_size": max(queue_sizes) if queue_sizes else 0,
            "min_queue_size": min(queue_sizes) if queue_sizes else 0,
            "avg_queue_size": sum(queue_sizes) / len(queue_sizes) if queue_sizes else 0,
            "unconditional_count": len(self.unconditional_queue),
        }
    
    def clear(self) -> None:
        """Clear all queues."""
        for q in self.queues:
            q.clear()
        self.unconditional_queue.clear()


class ClassAwareSampleQueue(SampleQueue):
    """
    Alias for SampleQueue for backward compatibility.
    
    This is the main class-aware sample queue as described in the paper.
    """
    pass
