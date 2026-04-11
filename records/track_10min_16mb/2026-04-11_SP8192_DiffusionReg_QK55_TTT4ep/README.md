# SP8192 + Diffusion Regularization Record Attempt

## Overview
This submission attempts to beat the 1.0810 BPB record by adding **text diffusion as a training regularization mechanism** to the optimized combination of techniques from previous top submissions.

## Key Innovations

### Text Diffusion Regularization (NEW)
Instead of simple noise injection, we implement proper **diffusion-based regularization** with:
- **Forward process**: Gradually add noise to token embeddings using a cosine beta schedule
- **Reverse process**: Lightweight denoiser network predicts clean embeddings from noisy inputs
- **Training objective**: MSE loss between denoised and original embeddings (weighted at 0.1)
- **Parameters**: Only ~35K additional parameters (64-dim hidden, 8 steps)

**Diffusion Hyperparameters:**
- `diffusion_enabled=1` - Enable diffusion regularization
- `diffusion_steps=8` - Number of diffusion timesteps
- `diffusion_beta_start=1e-4` - Start of noise schedule
- `diffusion_beta_end=0.02` - End of noise schedule  
- `diffusion_denoise_strength=0.85` - Learnable denoising strength
- `diffusion_hidden_dim=64` - Denoiser hidden dimension

### Combined Optimizations (from previous records)
Building on bigbag's 1.0810 BPB baseline:

| Parameter | Baseline | **Optimized** | Source |
|-----------|----------|---------------|--------|
| QK-Gain Init | 5.25 | **5.5** | Extended tuning |
| TTT Epochs | 3 | **4** | More adaptation |
| TTT Learning Rate | 0.005 | **0.006** | Faster convergence |
| MLP Multiplier | 4.0 | **4.25** | Enhanced capacity |
| Matrix Clip Sigmas | 12.85 | **13.2** | Better quantization |
| Embed Clip Sigmas | 20.0 | **21.5** | Embedding preservation |
| **Diffusion Reg** | N/A | **Enabled** | **NEW** |

## Architecture Summary

- **Vocabulary**: SP8192 (8192 tokens, int6 quantized)
- **Layers**: 11L with 3-layer depth recurrence (L3-5 looped 2x)
- **Dimensions**: 512d model, 8 heads, 4 KV heads
- **MLP**: 4.25x multiplier with parallel residuals from layer 7+
- **Optimization**: MuonEq-R + EMA 0.9965
- **Quantization**: GPTQ int6 with SD-Clip (matrix: 13.2σ, embed: 21.5σ)
- **TTT**: Legal score-first SGD, 4 epochs, LR=0.006
- **Diffusion**: 8-step DDPM-style regularization on embeddings

## Expected Improvements

| Component | Estimated BPB Gain |
|-----------|-------------------|
| QK-Gain 5.5 vs 5.25 | ~0.0003 |
| TTT 4ep vs 3ep | ~0.0005 |
| TTT LR 0.006 vs 0.005 | ~0.0002 |
| MLP mult 4.25 vs 4.0 | ~0.0003 |
| Quantization tweaks | ~0.0002 |
| **Diffusion regularization** | **~0.0004-0.0006** |
| **Total expected** | **~0.0019-0.0021** |

**Target BPB: ~1.0789-1.0791** (vs 1.0810 baseline)

## How Diffusion Helps

1. **Embedding Smoothing**: Denoising forces embeddings to lie on a smoother manifold
2. **Regularization**: Prevents overfitting during aggressive 10-minute training
3. **Quantization Resilience**: Smoother embeddings survive int6 quantization better
4. **Generalization**: Similar to dropout but with structured noise from diffusion process

## Compliance

- ✅ **Artifact Size**: <16MB (diffusion adds only ~35KB)
- ✅ **Training Time**: <10 minutes on 8xH100 (minimal overhead from diffusion)
- ✅ **Legal Techniques**: All methods from published records + standard diffusion

## Files

- `train_gpt_optimized.py` - Full uncompressed training script with diffusion
- `train_gpt.py` - Compressed version for submission (after training)
- `submission.json` - Metadata and compliance info

## Usage

```bash
export DIFFUSION_ENABLED=1
export DIFFUSION_STEPS=8
export DIFFUSION_BETA_START=1e-4
export DIFFUSION_BETA_END=0.02
export DIFFUSION_DENOISE_STRENGTH=0.85
export DIFFUSION_HIDDEN_DIM=64

# Run with other standard hyperparameters
python3 train_gpt_optimized.py
```

## References

- bigbag's 1.0810 BPB record (2026-04-09)
- DDPM: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- Previous Parameter Golf submissions for architecture baselines
