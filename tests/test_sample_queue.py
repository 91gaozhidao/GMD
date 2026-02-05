"""
Tests for Sample Queue.
"""

import pytest
import torch

from drifting.data.sample_queue import SampleQueue, GlobalSampleQueue


class TestSampleQueue:
    """Tests for SampleQueue class."""
    
    def test_initialization(self):
        """Test that queue initializes correctly."""
        queue = SampleQueue(queue_size=64, num_classes=10)
        assert len(queue.queues) == 11  # 10 classes + 1 unconditional
    
    def test_add_samples(self):
        """Test adding samples to queue."""
        queue = SampleQueue(queue_size=64, num_classes=10, latent_shape=(4, 8, 8))
        
        samples = torch.randn(16, 4, 8, 8)
        labels = torch.randint(0, 10, (16,))
        
        queue.add(samples, labels)
        
        stats = queue.stats()
        assert stats['total_samples'] == 16
    
    def test_sample_from_queue(self):
        """Test sampling from queue."""
        queue = SampleQueue(queue_size=64, num_classes=10, latent_shape=(4, 8, 8))
        
        # Add samples
        samples = torch.randn(32, 4, 8, 8)
        labels = torch.zeros(32, dtype=torch.long)  # All class 0
        queue.add(samples, labels)
        
        # Sample back
        query_labels = torch.zeros(8, dtype=torch.long)
        sampled = queue.sample(query_labels, num_samples=1)
        
        assert sampled.shape == (8, 4, 8, 8)
    
    def test_fifo_behavior(self):
        """Test FIFO queue behavior."""
        queue = SampleQueue(queue_size=4, num_classes=2, latent_shape=(2,))
        
        # Add samples
        for i in range(10):
            samples = torch.full((1, 2), float(i))
            labels = torch.zeros(1, dtype=torch.long)
            queue.add(samples, labels)
        
        # Queue should only keep last 4 samples
        assert len(queue.queues[0]) == 4
    
    def test_is_ready(self):
        """Test is_ready check."""
        queue = SampleQueue(queue_size=64, num_classes=10, latent_shape=(4, 8, 8))
        
        assert not queue.is_ready(min_samples=32)
        
        samples = torch.randn(64, 4, 8, 8)
        labels = torch.randint(0, 10, (64,))
        queue.add(samples, labels)
        
        assert queue.is_ready(min_samples=32)
    
    def test_unconditional_queue(self):
        """Test unconditional sample queue."""
        queue = SampleQueue(queue_size=64, num_classes=10, latent_shape=(4, 8, 8))
        
        samples = torch.randn(16, 4, 8, 8)
        queue.add_unconditional(samples)
        
        sampled = queue.sample_unconditional(8)
        assert sampled.shape == (8, 4, 8, 8)
    
    def test_empty_queue_fallback(self):
        """Test sampling from empty queue returns random samples."""
        queue = SampleQueue(queue_size=64, num_classes=10, latent_shape=(4, 8, 8))
        
        query_labels = torch.zeros(4, dtype=torch.long)
        sampled = queue.sample(query_labels, num_samples=1)
        
        # Should return random samples of correct shape
        assert sampled.shape == (4, 4, 8, 8)
    
    def test_stats(self):
        """Test statistics reporting."""
        queue = SampleQueue(queue_size=64, num_classes=10, latent_shape=(4, 8, 8))
        
        samples = torch.randn(32, 4, 8, 8)
        labels = torch.randint(0, 5, (32,))  # Only use first 5 classes
        queue.add(samples, labels)
        
        stats = queue.stats()
        assert 'total_samples' in stats
        assert 'num_non_empty' in stats
        assert stats['num_non_empty'] <= 5  # At most 5 non-empty queues


class TestGlobalSampleQueue:
    """Tests for GlobalSampleQueue class."""
    
    def test_initialization(self):
        """Test that queue initializes correctly."""
        queue = GlobalSampleQueue(queue_size=64, latent_shape=(4, 8, 8))
        assert len(queue) == 0
    
    def test_add_and_sample(self):
        """Test adding and sampling."""
        queue = GlobalSampleQueue(queue_size=64, latent_shape=(4, 8, 8))
        
        samples = torch.randn(32, 4, 8, 8)
        queue.add(samples)
        
        assert len(queue) == 32
        
        sampled = queue.sample(8)
        assert sampled.shape == (8, 4, 8, 8)
    
    def test_fifo_behavior(self):
        """Test FIFO queue behavior."""
        queue = GlobalSampleQueue(queue_size=4, latent_shape=(2,))
        
        for i in range(10):
            samples = torch.full((1, 2), float(i))
            queue.add(samples)
        
        assert len(queue) == 4
    
    def test_is_ready(self):
        """Test is_ready check."""
        queue = GlobalSampleQueue(queue_size=64, latent_shape=(4, 8, 8))
        
        assert not queue.is_ready(min_samples=32)
        
        samples = torch.randn(64, 4, 8, 8)
        queue.add(samples)
        
        assert queue.is_ready(min_samples=32)
