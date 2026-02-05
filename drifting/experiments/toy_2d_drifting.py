"""
2D Toy Problem: Mathematical Verification of Drifting Field Theory

This script verifies the "Drifting Field" equilibrium theory by reproducing
the 2D Gaussian mixture experiment (Figure 3 from the paper).

The core hypothesis is that an anti-symmetric field V(x) forces the generated 
distribution to match the data distribution. If this logic works in 2D, it 
provides confidence for the deep learning model.

Usage:
    python -m drifting.experiments.toy_2d_drifting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import os

from ..utils.drifting_field import compute_V, drifting_loss


class SimpleMLP(nn.Module):
    """Simple MLP generator for 2D toy problem."""
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        output_dim: int = 2,
        num_layers: int = 4,
    ):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def create_bimodal_gaussian(
    n_samples: int,
    centers: List[Tuple[float, float]] = [(-2, 0), (2, 0)],
    std: float = 0.5,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Create samples from a bimodal 2D Gaussian distribution.
    
    Args:
        n_samples: Number of samples to generate
        centers: List of (x, y) centers for each mode
        std: Standard deviation for each mode
        weights: Mixing weights (default: uniform)
        
    Returns:
        Samples of shape (n_samples, 2)
    """
    if weights is None:
        weights = [1.0 / len(centers)] * len(centers)
    
    n_modes = len(centers)
    samples_per_mode = []
    
    for i, (cx, cy) in enumerate(centers):
        n_mode = int(n_samples * weights[i])
        if i == n_modes - 1:
            # Handle rounding
            n_mode = n_samples - sum(len(s) for s in samples_per_mode)
        
        samples = torch.randn(n_mode, 2) * std + torch.tensor([[cx, cy]])
        samples_per_mode.append(samples)
    
    all_samples = torch.cat(samples_per_mode, dim=0)
    
    # Shuffle
    perm = torch.randperm(all_samples.shape[0])
    return all_samples[perm]


def train_toy_drifting(
    num_iterations: int = 5000,
    batch_size: int = 256,
    queue_size: int = 512,
    sigma: float = 0.5,
    temperature: float = 1.0,
    lr: float = 1e-3,
    device: str = "cpu",
    save_path: Optional[str] = None,
    visualize_every: int = 500,
) -> SimpleMLP:
    """
    Train a 2D generator using the drifting field method.
    
    Args:
        num_iterations: Number of training iterations
        batch_size: Batch size for training
        queue_size: Size of the sample queue for positive samples
        sigma: Kernel bandwidth
        temperature: Softmax temperature
        lr: Learning rate
        device: Device to train on
        save_path: Path to save visualizations
        visualize_every: How often to save visualizations
        
    Returns:
        Trained generator model
    """
    # Create model
    generator = SimpleMLP(input_dim=2, hidden_dim=256, output_dim=2).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    
    # Create target distribution (bimodal Gaussian)
    target_centers = [(-2, 0), (2, 0)]
    target_std = 0.5
    
    # Initialize sample queue with real samples
    sample_queue = create_bimodal_gaussian(
        queue_size, centers=target_centers, std=target_std
    ).to(device)
    
    # Training history
    losses = []
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    print("Training 2D Drifting Model...")
    print(f"Target: Bimodal Gaussian with centers {target_centers}")
    
    for iteration in range(num_iterations):
        # Sample noise
        z = torch.randn(batch_size, 2, device=device)
        
        # Generate samples
        x_gen = generator(z)
        
        # Sample from queue (positive samples)
        indices = torch.randint(0, queue_size, (batch_size,))
        x_data = sample_queue[indices]
        
        # Compute drifting field V(x)
        V = compute_V(
            x_gen, x_data, x_gen,
            sigma=sigma,
            temperature=temperature,
            normalize_features=False,  # 2D is already low-dim
            normalize_drift=True,
        )
        
        # Compute target with stop-gradient
        target = (x_gen + V).detach()
        
        # Compute loss
        loss = drifting_loss(x_gen, V, target)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Refresh sample queue with new real samples
        new_samples = create_bimodal_gaussian(
            batch_size, centers=target_centers, std=target_std
        ).to(device)
        
        # FIFO update: remove oldest, add newest
        sample_queue = torch.cat([sample_queue[batch_size:], new_samples], dim=0)
        
        # Logging and visualization
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item():.6f}")
        
        if save_path and (iteration + 1) % visualize_every == 0:
            visualize_progress(
                generator, target_centers, target_std, 
                iteration + 1, losses, save_path, device
            )
    
    # Final visualization
    if save_path:
        visualize_progress(
            generator, target_centers, target_std,
            num_iterations, losses, save_path, device, final=True
        )
    
    return generator


def visualize_progress(
    generator: SimpleMLP,
    target_centers: List[Tuple[float, float]],
    target_std: float,
    iteration: int,
    losses: List[float],
    save_path: str,
    device: str,
    final: bool = False,
):
    """Visualize training progress."""
    generator.eval()
    
    with torch.no_grad():
        # Generate samples
        z = torch.randn(1000, 2, device=device)
        x_gen = generator(z).cpu().numpy()
        
        # Real samples
        x_real = create_bimodal_gaussian(
            1000, centers=target_centers, std=target_std
        ).numpy()
    
    generator.train()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot real distribution
    axes[0].scatter(x_real[:, 0], x_real[:, 1], alpha=0.5, s=10, c='blue')
    axes[0].set_title('Target Distribution')
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)
    axes[0].set_aspect('equal')
    
    # Plot generated distribution
    axes[1].scatter(x_gen[:, 0], x_gen[:, 1], alpha=0.5, s=10, c='red')
    axes[1].set_title(f'Generated Distribution (Iter {iteration})')
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-5, 5)
    axes[1].set_aspect('equal')
    
    # Plot loss curve
    axes[2].plot(losses)
    axes[2].set_title('Training Loss')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Loss')
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    
    filename = f"progress_iter_{iteration}.png" if not final else "final_result.png"
    plt.savefig(os.path.join(save_path, filename), dpi=150)
    plt.close()


def visualize_drifting_field(
    x_data: torch.Tensor,
    x_gen: torch.Tensor,
    sigma: float = 0.5,
    temperature: float = 1.0,
    grid_resolution: int = 20,
    save_path: Optional[str] = None,
):
    """
    Visualize the drifting field as a vector field.
    
    Args:
        x_data: Data samples
        x_gen: Generated samples
        sigma: Kernel bandwidth
        temperature: Softmax temperature
        grid_resolution: Resolution of the grid
        save_path: Path to save the figure
    """
    # Create grid
    x = np.linspace(-4, 4, grid_resolution)
    y = np.linspace(-4, 4, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid points
    grid_points = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=-1),
        dtype=torch.float32
    )
    
    # Compute drifting field at grid points
    V = compute_V(
        grid_points, x_data, x_gen,
        sigma=sigma,
        temperature=temperature,
        normalize_features=False,
        normalize_drift=False,
    )
    
    V_np = V.numpy()
    U = V_np[:, 0].reshape(grid_resolution, grid_resolution)
    W = V_np[:, 1].reshape(grid_resolution, grid_resolution)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot vector field
    ax.quiver(X, Y, U, W, alpha=0.6)
    
    # Plot data and generated samples
    x_data_np = x_data.numpy()
    x_gen_np = x_gen.numpy()
    ax.scatter(x_data_np[:, 0], x_data_np[:, 1], c='blue', s=30, alpha=0.7, label='Data')
    ax.scatter(x_gen_np[:, 0], x_gen_np[:, 1], c='red', s=30, alpha=0.7, label='Generated')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Drifting Field V(x)')
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    """Run the 2D toy experiment."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = "outputs/toy_2d"
    os.makedirs(output_dir, exist_ok=True)
    
    # First, visualize the drifting field before training
    print("\nVisualizing initial drifting field...")
    x_data = create_bimodal_gaussian(100, centers=[(-2, 0), (2, 0)], std=0.5)
    x_gen = torch.randn(100, 2) * 2  # Random initial positions
    
    visualize_drifting_field(
        x_data, x_gen,
        sigma=0.5,
        temperature=1.0,
        save_path=os.path.join(output_dir, "drifting_field_initial.png")
    )
    
    # Train the model
    print("\nStarting training...")
    generator = train_toy_drifting(
        num_iterations=5000,
        batch_size=256,
        queue_size=512,
        sigma=0.5,
        temperature=1.0,
        lr=1e-3,
        device=device,
        save_path=output_dir,
        visualize_every=1000,
    )
    
    # Final evaluation
    print("\nGenerating final samples...")
    generator.eval()
    with torch.no_grad():
        z = torch.randn(1000, 2, device=device)
        x_gen = generator(z).cpu()
    
    # Visualize final drifting field
    x_data = create_bimodal_gaussian(100, centers=[(-2, 0), (2, 0)], std=0.5)
    visualize_drifting_field(
        x_data, x_gen[:100],
        sigma=0.5,
        temperature=1.0,
        save_path=os.path.join(output_dir, "drifting_field_final.png")
    )
    
    print(f"\nResults saved to {output_dir}/")
    print("Experiment complete!")


if __name__ == "__main__":
    main()
