# Generative Modeling via Drifting

A PyTorch implementation of the "Generative Modeling via Drifting" paper. This method differs fundamentally from Diffusion or Flow Matching because it evolves the *pushforward distribution* during training via a "drifting field," allowing for **one-step (1-NFE) inference**.

## Overview

The drifting field method is based on an elegant equilibrium theory:
- **V(x) = V⁺(x) - V⁻(x)**: The drifting field has two components
  - **V⁺(x)**: Attraction from data samples (positive)
  - **V⁻(x)**: Repulsion from generated samples (negative)
- At equilibrium, when the generated distribution matches the data distribution, V(x) → 0.

### Key Features
- **One-Step Inference**: Unlike diffusion models that require many steps, this method generates samples in a single forward pass
- **Training-Time CFG**: Classifier-Free Guidance is baked into the training objective
- **Multi-Scale Loss**: Computed at multiple feature extraction scales for better quality
- **DiT Architecture**: Uses Diffusion Transformer with SwiGLU, RoPE, QK-Norm, and AdaLN-Zero

## Installation

```bash
# Clone the repository
git clone https://github.com/91gaozhidao/Generative-Modeling-via-Drifting.git
cd Generative-Modeling-via-Drifting

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Phase 1: 2D Toy Problem (Mathematical Verification)

Verify the drifting field theory with a 2D Gaussian mixture:

```python
# Run the 2D toy experiment
python -m drifting.experiments.toy_2d_drifting
```

This will:
1. Visualize the drifting field
2. Train a simple MLP to generate bimodal Gaussian samples
3. Save progress visualizations to `outputs/toy_2d/`

### Phase 2: Using the DiT Generator

```python
import torch
from drifting.models import DriftingDiT

# Create model (for 32x32x4 latent space)
model = DriftingDiT(
    img_size=32,
    patch_size=2,
    in_chans=4,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12,
)

# Generate samples (one-step inference!)
labels = torch.randint(0, 1000, (8,))
samples = model.generate(batch_size=8, y=labels, cfg_scale=1.5, device='cuda')
```

### Phase 3: Drifting Loss

```python
from drifting.models import DriftingLoss

# Create loss module
loss_fn = DriftingLoss(
    feature_extractor='latent',  # or 'resnet18' for pretrained
    temperatures=[0.1, 0.5, 1.0, 2.0],
    normalize_features=True,
    normalize_drift=True,
)

# Compute loss
loss = loss_fn(x_generated, x_real)
```

### Phase 4 & 5: Full Training with CFG

```python
from drifting.models import DriftingDiT
from drifting.training import DriftingTrainer, create_trainer

# Create model
model = DriftingDiT(num_classes=1000)

# Create trainer with all components
trainer = create_trainer(
    model=model,
    feature_extractor='latent',
    learning_rate=1e-4,
    device='cuda',
    queue_size=128,
    cfg_scale_range=(1.0, 7.5),
    uncond_prob=0.1,
)

# Training loop
for epoch in range(num_epochs):
    metrics = trainer.train_epoch(dataloader, use_cfg_training=True)
    print(f"Epoch {epoch}: Loss = {metrics['avg_loss']:.4f}")
    
    # One-step generation
    samples = trainer.generate(
        batch_size=16,
        labels=class_labels,
        cfg_scale=2.0,
    )
```

## Architecture

### DriftingDiT Components

1. **SwiGLU Activation**: `SwiGLU(x) = Swish(xW₁) ⊗ (xW₂)`
2. **RoPE**: Rotary Positional Embedding for attention
3. **QK-Norm**: LayerNorm on queries and keys for training stability
4. **AdaLN-Zero**: Adaptive layer normalization with zero initialization
5. **Style Tokens**: 32 learnable tokens for diversity beyond Gaussian noise

### Model Variants

| Model | Embed Dim | Depth | Heads | Parameters |
|-------|-----------|-------|-------|------------|
| Small | 384 | 6 | 6 | ~25M |
| Base | 768 | 12 | 12 | ~130M |
| Large | 1024 | 24 | 16 | ~460M |
| XL | 1152 | 28 | 16 | ~675M |

## Core Algorithm

The drifting field V(x) is computed as:

```python
def compute_V(x, x_data, x_gen, sigma, temperature):
    # Compute kernel matrices
    K_data = exp(-||x - x_data||² / (2σ²))
    K_gen = exp(-||x - x_gen||² / (2σ²))
    
    # Softmax normalization (both axes)
    W_data = softmax_normalize_2d(K_data, temperature)
    W_gen = softmax_normalize_2d(K_gen, temperature)
    
    # Attraction to data
    V_plus = W_data @ x_data - x
    
    # Repulsion from generated
    V_minus = W_gen @ x_gen - x
    
    # Combined field
    V = V_plus - V_minus
    
    return V
```

The training loss with stop-gradient:

```python
target = stop_gradient(x_gen + V(x_gen))
loss = MSE(x_gen, target)
```

## Project Structure

```
drifting/
├── __init__.py
├── models/
│   ├── drifting_dit.py      # DiT architecture
│   ├── drifting_loss.py     # Loss computation
│   └── feature_extractor.py # Multi-scale features
├── utils/
│   └── drifting_field.py    # Core V(x) computation
├── data/
│   └── sample_queue.py      # FIFO sample queue
├── training/
│   └── __init__.py          # Training loop + CFG
└── experiments/
    └── toy_2d_drifting.py   # 2D verification
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_drifting_field.py -v

# Run with coverage
pytest tests/ --cov=drifting --cov-report=html
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{drifting2024,
  title={Generative Modeling via Drifting},
  author={...},
  journal={...},
  year={2024}
}
```

## License

MIT License