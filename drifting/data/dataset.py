"""
Dataset Module for Drifting Field Generative Models

This module provides dataset classes for loading pre-cached latent representations,
enabling efficient training without VAE encoding during the training loop.

Key classes:
- LatentDataset: Load pre-cached latent files (batched or individual)
- LatentDatasetIndividual: Load individually saved latent files
- ClassGroupedBatchSampler: Sampler for class-grouped batches (Paper requirement)
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Union, Iterator
import json

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np


class LatentDataset(Dataset):
    """
    Dataset for loading pre-cached VAE latent representations.
    
    This dataset loads latents that were pre-computed and saved by cache_latents.py,
    enabling efficient training without VAE encoding during the training loop.
    
    Supports two storage formats:
    1. Batched: Single latents.pt and labels.pt files (default)
    2. Individual: Separate .pt files for each sample
    
    Args:
        data_dir: Directory containing cached latents
        transform: Optional transform to apply to latents (e.g., augmentation)
        load_in_memory: Whether to load all latents into memory (default: True)
        subset_size: Optional limit on number of samples to use
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[callable] = None,
        load_in_memory: bool = True,
        subset_size: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.load_in_memory = load_in_memory
        
        # Check if metadata exists
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Determine storage format
        self.individual_storage = self.metadata.get('save_individually', False)
        
        if self.individual_storage:
            # Individual file storage
            self.sample_files = sorted(self.data_dir.glob("sample_*.pt"))
            self.num_samples = len(self.sample_files)
            self.latents = None
            self.labels = None
        else:
            # Batched storage
            latents_path = self.data_dir / "latents.pt"
            labels_path = self.data_dir / "labels.pt"
            
            if not latents_path.exists():
                raise FileNotFoundError(f"Latents file not found: {latents_path}")
            if not labels_path.exists():
                raise FileNotFoundError(f"Labels file not found: {labels_path}")
            
            if load_in_memory:
                self.latents = torch.load(latents_path, weights_only=True)
                self.labels = torch.load(labels_path, weights_only=True)
            else:
                # Memory-map for large datasets
                self.latents_path = latents_path
                self.labels_path = labels_path
                self.latents = None
                self.labels = None
                # Load labels for length calculation
                self._labels_cache = torch.load(labels_path, weights_only=True)
            
            self.num_samples = (
                len(self.latents) if self.latents is not None 
                else len(self._labels_cache)
            )
        
        # Apply subset if specified
        if subset_size is not None:
            self.num_samples = min(self.num_samples, subset_size)
        
        # Get latent shape from metadata or first sample
        self.latent_shape = self._get_latent_shape()
    
    def _get_latent_shape(self) -> Tuple[int, ...]:
        """Get the shape of a single latent tensor."""
        if 'latent_shape' in self.metadata:
            return tuple(self.metadata['latent_shape'])
        
        # Infer from data
        if self.individual_storage and self.sample_files:
            sample = torch.load(self.sample_files[0], weights_only=True)
            return tuple(sample['latent'].shape)
        elif self.latents is not None:
            return tuple(self.latents.shape[1:])
        else:
            # Load first sample to get shape
            latents = torch.load(self.latents_path, weights_only=True)
            return tuple(latents.shape[1:])
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single latent sample and its label.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (latent_tensor, class_label)
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        
        if self.individual_storage:
            # Load from individual file
            sample = torch.load(self.sample_files[idx], weights_only=True)
            latent = sample['latent']
            label = sample['label']
        else:
            if self.latents is not None:
                # Direct access from memory
                latent = self.latents[idx]
                label = self.labels[idx].item()
            else:
                # Load from disk
                latents = torch.load(self.latents_path, weights_only=True)
                labels = torch.load(self.labels_path, weights_only=True)
                latent = latents[idx]
                label = labels[idx].item()
        
        # Apply transform if provided
        if self.transform is not None:
            latent = self.transform(latent)
        
        return latent, label
    
    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return self.metadata.get('num_classes', 1000)


class LatentDatasetIndividual(Dataset):
    """
    Dataset for loading individually saved latent files.
    
    This is optimized for very large datasets where loading all latents
    into memory is not feasible.
    
    Args:
        data_dir: Directory containing individual .pt files
        transform: Optional transform to apply to latents
        file_pattern: Glob pattern for sample files (default: "sample_*.pt")
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[callable] = None,
        file_pattern: str = "sample_*.pt",
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Find all sample files
        self.sample_files = sorted(self.data_dir.glob(file_pattern))
        
        if not self.sample_files:
            raise FileNotFoundError(
                f"No sample files found in {data_dir} matching pattern {file_pattern}"
            )
        
        self.num_samples = len(self.sample_files)
        
        # Get latent shape from first sample
        first_sample = torch.load(self.sample_files[0], weights_only=True)
        self.latent_shape = tuple(first_sample['latent'].shape)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single latent sample and its label.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (latent_tensor, class_label)
        """
        sample = torch.load(self.sample_files[idx], weights_only=True)
        latent = sample['latent']
        label = sample['label']
        
        if self.transform is not None:
            latent = self.transform(latent)
        
        return latent, label


class DummyLatentDataset(Dataset):
    """
    Dummy latent dataset for testing and debugging.
    
    Generates random latent tensors without requiring pre-cached data.
    
    Args:
        num_samples: Number of samples
        num_classes: Number of classes
        latent_shape: Shape of latent tensors (default: (4, 32, 32))
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 1000,
        latent_shape: Tuple[int, ...] = (4, 32, 32),
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self._num_classes = num_classes
        self.latent_shape = latent_shape
        self.seed = seed
        
        # Generate fixed random labels
        rng = np.random.RandomState(seed)
        self.labels = rng.randint(0, num_classes, size=num_samples)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Generate a random latent tensor.
        
        Returns:
            Tuple of (latent_tensor, label)
        """
        # Use deterministic seed for reproducibility
        rng = np.random.RandomState(self.seed + idx)
        latent = rng.randn(*self.latent_shape).astype(np.float32)
        
        return torch.from_numpy(latent), int(self.labels[idx])
    
    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return self._num_classes


class ClassGroupedBatchSampler(Sampler):
    """
    Class-Grouped Batch Sampler for Drifting Field training.
    
    This sampler enforces a specific batch structure where each batch contains
    samples from a small number of classes, with many samples per class.
    
    This is crucial for single-GPU training because Drifting Field estimation
    requires a high number of negative samples per class within a batch.
    Random sampling on a single GPU (Batch=128) fails because samples per class << 1.
    
    Per Paper requirements:
    - Select K classes per batch (e.g., K=4)
    - Sample M images per class (e.g., M=32)
    - Total Batch Size = K * M (e.g., 128)
    
    This ensures accurate drift estimation for the selected classes in each step.
    
    Args:
        dataset: Dataset with labels accessible
        num_classes_per_batch: Number of classes to include per batch (K)
        samples_per_class: Number of samples per class (M)
        num_classes: Total number of classes in dataset
        drop_last: Whether to drop incomplete batches
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        dataset: Dataset,
        num_classes_per_batch: int = 4,
        samples_per_class: int = 32,
        num_classes: int = 1000,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.num_classes_per_batch = num_classes_per_batch
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.drop_last = drop_last
        self.batch_size = num_classes_per_batch * samples_per_class
        
        # Random state for reproducibility
        self.rng = np.random.RandomState(seed)
        
        # Build class-to-indices mapping
        self.class_indices = self._build_class_indices()
        
        # Calculate number of batches
        self._num_batches = self._calculate_num_batches()
    
    def _build_class_indices(self) -> dict:
        """Build a mapping from class labels to sample indices."""
        class_indices = {c: [] for c in range(self.num_classes)}
        
        # Iterate through dataset to get labels
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            if 0 <= label < self.num_classes:
                class_indices[label].append(idx)
        
        return class_indices
    
    def _calculate_num_batches(self) -> int:
        """Calculate the number of batches based on available samples."""
        # Count classes with enough samples
        valid_classes = [
            c for c, indices in self.class_indices.items()
            if len(indices) >= self.samples_per_class
        ]
        
        if len(valid_classes) < self.num_classes_per_batch:
            # Fall back to classes with any samples
            valid_classes = [c for c, indices in self.class_indices.items() if len(indices) > 0]
        
        # Estimate batches based on total samples
        total_samples = sum(len(self.class_indices[c]) for c in valid_classes)
        return max(1, total_samples // self.batch_size)
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches of indices with class grouping."""
        # Get classes with samples
        available_classes = [
            c for c, indices in self.class_indices.items()
            if len(indices) > 0
        ]
        
        if len(available_classes) == 0:
            return
        
        # Shuffle available classes
        self.rng.shuffle(available_classes)
        
        # Create local copy of indices for sampling without replacement within epoch
        class_indices_copy = {c: list(indices) for c, indices in self.class_indices.items()}
        for indices in class_indices_copy.values():
            self.rng.shuffle(indices)
        
        batch_count = 0
        while batch_count < self._num_batches:
            batch = []
            
            # Select K random classes for this batch
            if len(available_classes) >= self.num_classes_per_batch:
                selected_classes = self.rng.choice(
                    available_classes, 
                    size=self.num_classes_per_batch, 
                    replace=False
                )  # Keep as numpy array for efficiency
            else:
                # If not enough classes, sample with replacement
                selected_classes = self.rng.choice(
                    available_classes, 
                    size=self.num_classes_per_batch, 
                    replace=True
                )  # Keep as numpy array for efficiency
            
            # Sample M indices from each selected class
            for cls in selected_classes:
                cls = int(cls)  # Convert numpy int to Python int once
                indices = class_indices_copy[cls]
                
                if len(indices) >= self.samples_per_class:
                    # Sample without replacement
                    sampled = indices[:self.samples_per_class]
                    class_indices_copy[cls] = indices[self.samples_per_class:]
                    batch.extend(sampled)
                else:
                    # Sample with replacement if not enough
                    sampled = self.rng.choice(
                        self.class_indices[cls], 
                        size=self.samples_per_class, 
                        replace=True
                    )  # Keep as numpy array
                    batch.extend(sampled.tolist())  # Convert only when extending
                
                # Refill if exhausted
                if len(class_indices_copy[cls]) < self.samples_per_class:
                    class_indices_copy[cls] = list(self.class_indices[cls])
                    self.rng.shuffle(class_indices_copy[cls])
            
            if len(batch) == self.batch_size:
                yield batch
                batch_count += 1
            elif not self.drop_last and len(batch) > 0:
                yield batch
                batch_count += 1
    
    def __len__(self) -> int:
        """Return the number of batches."""
        return self._num_batches


def create_dataloader(
    data_dir: str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    use_class_grouped_sampler: bool = False,
    num_classes_per_batch: int = 4,
    samples_per_class: int = 32,
    num_classes: int = 1000,
    seed: int = 42,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for cached latent data.
    
    Args:
        data_dir: Directory containing cached latents
        batch_size: Batch size (ignored if use_class_grouped_sampler=True)
        shuffle: Whether to shuffle data (ignored if use_class_grouped_sampler=True)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        use_class_grouped_sampler: Whether to use class-grouped batch sampling
                                   (recommended for single-GPU training)
        num_classes_per_batch: Number of classes per batch (K) when using grouped sampler
        samples_per_class: Samples per class (M) when using grouped sampler
        num_classes: Total number of classes in dataset
        seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to LatentDataset
        
    Returns:
        Configured DataLoader
    """
    dataset = LatentDataset(data_dir, **kwargs)
    
    if use_class_grouped_sampler:
        # Use class-grouped batch sampler for Drifting Field training
        batch_sampler = ClassGroupedBatchSampler(
            dataset=dataset,
            num_classes_per_batch=num_classes_per_batch,
            samples_per_class=samples_per_class,
            num_classes=num_classes,
            drop_last=drop_last,
            seed=seed,
        )
        
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )


def create_dummy_dataloader(
    num_samples: int = 1000,
    num_classes: int = 1000,
    batch_size: int = 64,
    latent_shape: Tuple[int, ...] = (4, 32, 32),
    seed: int = 42,
    use_class_grouped_sampler: bool = False,
    num_classes_per_batch: int = 4,
    samples_per_class: int = 32,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader with dummy latent data for testing.
    
    Args:
        num_samples: Number of dummy samples
        num_classes: Number of classes
        batch_size: Batch size (ignored if use_class_grouped_sampler=True)
        latent_shape: Shape of latent tensors
        seed: Random seed
        use_class_grouped_sampler: Whether to use class-grouped batch sampling
        num_classes_per_batch: Number of classes per batch when using grouped sampler
        samples_per_class: Samples per class when using grouped sampler
        **kwargs: Additional arguments passed to DataLoader
        
    Returns:
        Configured DataLoader with dummy data
    """
    dataset = DummyLatentDataset(
        num_samples=num_samples,
        num_classes=num_classes,
        latent_shape=latent_shape,
        seed=seed,
    )
    
    if use_class_grouped_sampler:
        batch_sampler = ClassGroupedBatchSampler(
            dataset=dataset,
            num_classes_per_batch=num_classes_per_batch,
            samples_per_class=samples_per_class,
            num_classes=num_classes,
            drop_last=kwargs.pop('drop_last', True),
            seed=seed,
        )
        
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=kwargs.pop('num_workers', 0),
            pin_memory=kwargs.pop('pin_memory', False),
            **kwargs,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=kwargs.pop('shuffle', True),
            num_workers=kwargs.pop('num_workers', 0),  # Use 0 for dummy data
            pin_memory=kwargs.pop('pin_memory', False),
            drop_last=kwargs.pop('drop_last', True),
            **kwargs,
        )
