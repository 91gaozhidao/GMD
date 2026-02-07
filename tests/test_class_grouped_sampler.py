"""
Tests for ClassGroupedBatchSampler.
"""

import pytest
import torch
import numpy as np
from collections import Counter

from drifting.data.dataset import (
    ClassGroupedBatchSampler, 
    DummyLatentDataset,
    create_dummy_dataloader,
)


class TestClassGroupedBatchSampler:
    """Tests for ClassGroupedBatchSampler."""
    
    def test_batch_size(self):
        """Test that batches have correct size K*M."""
        num_classes_per_batch = 4
        samples_per_class = 8
        expected_batch_size = num_classes_per_batch * samples_per_class
        
        dataset = DummyLatentDataset(
            num_samples=1000, num_classes=100, seed=42
        )
        
        sampler = ClassGroupedBatchSampler(
            dataset=dataset,
            num_classes_per_batch=num_classes_per_batch,
            samples_per_class=samples_per_class,
            num_classes=100,
            seed=42,
        )
        
        for batch_indices in sampler:
            assert len(batch_indices) == expected_batch_size
            break  # Just test first batch
    
    def test_class_grouping(self):
        """Test that batches contain samples from exactly K classes."""
        num_classes_per_batch = 4
        samples_per_class = 8
        num_classes = 20
        
        dataset = DummyLatentDataset(
            num_samples=200, num_classes=num_classes, seed=42
        )
        
        sampler = ClassGroupedBatchSampler(
            dataset=dataset,
            num_classes_per_batch=num_classes_per_batch,
            samples_per_class=samples_per_class,
            num_classes=num_classes,
            seed=42,
        )
        
        for batch_indices in sampler:
            # Get labels for the batch
            labels = [dataset.labels[idx] for idx in batch_indices]
            unique_classes = set(labels)
            
            # Should have at most num_classes_per_batch unique classes
            assert len(unique_classes) <= num_classes_per_batch
            break
    
    def test_samples_per_class_count(self):
        """Test that each class has approximately M samples in a batch."""
        num_classes_per_batch = 4
        samples_per_class = 8
        num_classes = 20
        
        dataset = DummyLatentDataset(
            num_samples=400, num_classes=num_classes, seed=42
        )
        
        sampler = ClassGroupedBatchSampler(
            dataset=dataset,
            num_classes_per_batch=num_classes_per_batch,
            samples_per_class=samples_per_class,
            num_classes=num_classes,
            seed=42,
        )
        
        for batch_indices in sampler:
            # Get labels for the batch
            labels = [dataset.labels[idx] for idx in batch_indices]
            class_counts = Counter(labels)
            
            # Each class should have samples_per_class samples
            for cls, count in class_counts.items():
                assert count == samples_per_class
            break
    
    def test_iter_yields_batches(self):
        """Test that sampler yields batches correctly."""
        dataset = DummyLatentDataset(
            num_samples=500, num_classes=50, seed=42
        )
        
        sampler = ClassGroupedBatchSampler(
            dataset=dataset,
            num_classes_per_batch=4,
            samples_per_class=8,
            num_classes=50,
            seed=42,
        )
        
        batch_count = 0
        for batch in sampler:
            assert isinstance(batch, list)
            assert all(isinstance(idx, (int, np.integer)) for idx in batch)
            batch_count += 1
            if batch_count > 5:
                break
        
        assert batch_count > 0
    
    def test_reproducibility(self):
        """Test that sampler produces same batches with same seed."""
        dataset = DummyLatentDataset(
            num_samples=200, num_classes=20, seed=42
        )
        
        sampler1 = ClassGroupedBatchSampler(
            dataset=dataset,
            num_classes_per_batch=4,
            samples_per_class=8,
            num_classes=20,
            seed=123,
        )
        
        sampler2 = ClassGroupedBatchSampler(
            dataset=dataset,
            num_classes_per_batch=4,
            samples_per_class=8,
            num_classes=20,
            seed=123,
        )
        
        # Get first batch from each
        batch1 = next(iter(sampler1))
        batch2 = next(iter(sampler2))
        
        assert batch1 == batch2
    
    def test_len(self):
        """Test that __len__ returns correct number of batches."""
        dataset = DummyLatentDataset(
            num_samples=1000, num_classes=100, seed=42
        )
        
        sampler = ClassGroupedBatchSampler(
            dataset=dataset,
            num_classes_per_batch=4,
            samples_per_class=8,
            num_classes=100,
            seed=42,
        )
        
        # Length should be positive
        assert len(sampler) > 0


class TestClassGroupedDataLoader:
    """Integration tests for DataLoader with ClassGroupedBatchSampler."""
    
    def test_dataloader_with_class_grouped_sampler(self):
        """Test that DataLoader works with class-grouped sampler."""
        dataloader = create_dummy_dataloader(
            num_samples=200,
            num_classes=20,
            batch_size=32,  # Ignored when using class-grouped sampler
            seed=42,
            use_class_grouped_sampler=True,
            num_classes_per_batch=4,
            samples_per_class=8,
        )
        
        for batch_latents, batch_labels in dataloader:
            # Check batch shape
            assert batch_latents.shape[0] == 4 * 8  # K * M
            
            # Check that we have limited classes
            unique_classes = torch.unique(batch_labels)
            assert len(unique_classes) <= 4
            break
    
    def test_dataloader_class_distribution(self):
        """Test that batches have balanced class distribution."""
        dataloader = create_dummy_dataloader(
            num_samples=500,
            num_classes=50,
            seed=42,
            use_class_grouped_sampler=True,
            num_classes_per_batch=4,
            samples_per_class=16,
        )
        
        for _, batch_labels in dataloader:
            label_counts = torch.bincount(batch_labels, minlength=50)
            non_zero_counts = label_counts[label_counts > 0]
            
            # Each present class should have samples_per_class samples
            for count in non_zero_counts:
                assert count == 16
            break
