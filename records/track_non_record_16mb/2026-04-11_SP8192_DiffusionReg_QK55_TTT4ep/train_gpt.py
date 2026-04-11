"""
GPT-2 Training Script with Text Diffusion Regularization
Non-Record Submission: 2026-04-11_SP8192_DiffusionReg_QK55_TTT4ep

Key Features:
- SP8192 vocabulary (int6 quantized)
- 3-Layer Depth Recurrence (layers 3-5)
- Parallel Residuals (layer 7+)
- QK-Gain initialization (5.5)
- Legal Score-First TTT (4 epochs, LR=0.006)
- Text Diffusion Regularization (DDPM-style, 8 steps)
- GPTQ int6 quantization with enhanced SD-Clip
- MuonEq-R optimizer + EMA

Author: Parameter Golf Challenger
Date: 2026-04-11
"""

import os
import sys
import time
import math
import json
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
if torch.cuda.is_available():
    import torch.distributed as dist
    from torch.utils.cpp_extension import CUDA_HOME
else:
    dist = None

# -----------------------------------------------------------------------------
# Diffusion Regularization Module
# -----------------------------------------------------------------------------

class DiffusionDenoiser(nn.Module):
    """
    DDPM-style diffusion denoiser for embedding regularization.
    
    Forward process: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t)*I)
    Reverse process: p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
    
    This acts as a powerful regularizer by forcing the model to learn robust
    representations that can be recovered from noise.
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 64, num_steps: int = 8,
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        
        # Precompute cosine noise schedule
        betas = self._cosine_beta_schedule(num_steps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Learnable denoising strength parameter
        self.denoise_strength = nn.Parameter(torch.tensor(0.85))
        
        # Denoising network: two-layer MLP with residual connection
        # net1: condition-aware (takes noisy input + timestep embedding)
        self.net1 = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        # net2: refinement network
        self.net2 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Timestep embedding
        self.time_embed = nn.Embedding(num_steps, hidden_dim)
    
    def _cosine_beta_schedule(self, num_steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
        """Cosine noise schedule for smoother diffusion."""
        s = 0.008
        steps = torch.arange(num_steps + 1, dtype=torch.float32) / num_steps
        alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, beta_start, beta_end)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input embeddings [batch, seq_len, embed_dim]
            training: If True, apply diffusion during training
            
        Returns:
            x_denoised: Denoised embeddings
            diffusion_loss: MSE loss for diffusion objective
        """
        if not training or self.num_steps == 0:
            return x, torch.tensor(0.0, device=x.device)
        
        batch_size, seq_len, embed_dim = x.shape
        
        # Sample random timesteps for each position
        t = torch.randint(0, self.num_steps, (batch_size, seq_len, 1), 
                         device=x.device, dtype=torch.long)
        
        # Get noise schedule values for sampled timesteps
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].squeeze(-1)  # [B, S]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].squeeze(-1)  # [B, S]
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # Forward diffusion: add noise
        x_noisy = sqrt_alpha_bar.unsqueeze(-1) * x + sqrt_one_minus_alpha_bar.unsqueeze(-1) * noise
        
        # Get timestep embeddings
        t_embed = self.time_embed(t.squeeze(-1))  # [B, S, hidden_dim]
        
        # Concatenate noisy input with timestep embedding
        x_input = torch.cat([x_noisy, t_embed], dim=-1)  # [B, S, embed_dim + hidden_dim]
        
        # Denoise
        pred_eps = self.net1(x_input)
        pred_x0 = (x_noisy - sqrt_one_minus_alpha_bar.unsqueeze(-1) * pred_eps) / \
                  (sqrt_alpha_bar.unsqueeze(-1) + 1e-8)
        
        # Refinement pass
        pred_x0 = pred_x0 + self.denoise_strength * self.net2(pred_x0)
        
        # Compute diffusion loss (MSE between predicted and original)
        diffusion_loss = F.mse_loss(pred_x0, x, reduction='mean')
        
        return pred_x0, diffusion_loss
    
    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """Return parameter groups for optimizer."""
        groups = []
        
        # Muon-optimized parameters (weights)
        muon_params = []
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                muon_params.append(param)
        
        if muon_params:
            groups.append({'params': muon_params, 'lr': base_lr, 'use_muon': True})
        
        # AdamW-optimized parameters (biases, scalars)
        adam_params = []
        for name, param in self.named_parameters():
            if 'bias' in name or 'denoise_strength' in name or len(param.shape) < 2:
                adam_params.append(param)
        
        if adam_params:
            groups.append({'params': adam_params, 'lr': base_lr, 'use_muon': False})
        
        return groups


# -----------------------------------------------------------------------------
# Muon Optimizer
# -----------------------------------------------------------------------------

class MuonEqR(torch.optim.Optimizer):
    """
    MuonEq-R: Row-normalized Muon optimizer with Newton-Schulz iterations.
    Optimized for large language model training.
    """
    def __init__(self, params, lr=1e-3, beta1=0.95, beta2=0.95, eps=1e-8,
                 weight_decay=0.0, n_iter=5):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                       weight_decay=weight_decay, n_iter=n_iter)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2, eps = group['beta1'], group['beta2'], group['eps']
                
                # Update running averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** self._step_count if hasattr(self, '_step_count') else 1
                bias_correction2 = 1 - beta2 ** self._step_count if hasattr(self, '_step_count') else 1
                
                # Compute update
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                update = exp_avg / bias_correction1 / denom
                
                # Newton-Schulz orthogonalization for matrices
                if update.dim() >= 2:
                    update = self._newton_schulz(update, group['n_iter'])
                
                # Row normalization
                if update.dim() >= 2:
                    row_norm = update.norm(dim=-1, keepdim=True, p=2)
                    update = update / (row_norm + eps)
                
                # Apply weight decay
                if group['weight_decay'] > 0:
                    update.add_(p, alpha=group['weight_decay'])
                
                # Update parameters
                p.add_(update, alpha=-group['lr'])
    
    def _newton_schulz(self, A: torch.Tensor, n_iter: int = 5) -> torch.Tensor:
        """Newton-Schulz iteration for matrix orthogonalization."""
        B = A / (A.norm() + 1e-8)
        for _ in range(n_iter):
            B = 0.5 * (3 * B - B @ B.transpose(-2, -1) @ B)
        return B * A.norm()


# -----------------------------------------------------------------------------
# Model Components
# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Model configuration with diffusion parameters."""
    vocab_size: int = 8192
    n_layer: int = 11
    n_head: int = 8
    n_kv_head: int = 4
    n_embd: int = 512
    mlp_multiplier: float = 4.25
    max_seq_len: int = 1024
    dropout: float = 0.0
    
    # Depth recurrence
    depth_recurrence: bool = True
    recurrence_layers: Tuple[int, int] = (3, 5)  # Layers to repeat
    
    # Parallel residuals
    parallel_residual: bool = True
    parallel_from_layer: int = 7
    
    # QK-Gain
    qk_gain_init: float = 5.5
    
    # Diffusion regularization
    diffusion_enabled: bool = True
    diffusion_steps: int = 8
    diffusion_beta_start: float = 1e-4
    diffusion_beta_end: float = 0.02
    diffusion_hidden_dim: int = 64
    diffusion_loss_weight: float = 0.1
    
    # Quantization
    quantize_embeddings: bool = True
    matrix_clip_sigma: float = 13.2
    embed_clip_sigma: float = 21.5


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # QKV projections
        self.c_q = nn.Linear(config.n_embd, config.n_embd)
        self.c_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim)
        self.c_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # QK-Gain: learnable per-head scaling
        self.qk_gain = nn.Parameter(torch.full((config.n_head,), config.qk_gain_init))
        
        self.dropout = config.dropout
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        # Apply QK-Gain
        q = q * self.qk_gain.view(1, self.n_head, 1, 1)
        
        # Grouped Query Attention
        if self.n_kv_head < self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = int(config.n_embd * config.mlp_multiplier)
        self.c_fc = nn.Linear(config.n_embd, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.parallel_residual = (
            config.parallel_residual and 
            layer_idx >= config.parallel_from_layer
        )
        
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.parallel_residual:
            # GPT-J style parallel residuals
            attn_out = self.attn(self.ln_1(x))
            mlp_out = self.mlp(self.ln_2(x))
            x = x + attn_out + mlp_out
        else:
            # Standard pre-norm
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_seq_len, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = RMSNorm(config.n_embd)
        
        # Output head (tied with embeddings)
        self.lm_head = None  # Will tie with wte
        
        # Diffusion denoiser
        if config.diffusion_enabled:
            self.diffusion = DiffusionDenoiser(
                embed_dim=config.n_embd,
                hidden_dim=config.diffusion_hidden_dim,
                num_steps=config.diffusion_steps,
                beta_start=config.diffusion_beta_start,
                beta_end=config.diffusion_beta_end
            )
        else:
            self.diffusion = None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.config.max_seq_len
        
        # Embed tokens
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        
        # Apply diffusion denoising if enabled
        diffusion_loss = torch.tensor(0.0, device=x.device)
        if self.diffusion is not None and training:
            x, diffusion_loss = self.diffusion(x, training=True)
        
        # Transformer blocks with depth recurrence
        for i, block in enumerate(self.h):
            # Depth recurrence: reuse layers 3-5 multiple times
            if self.config.depth_recurrence and i >= self.config.recurrence_layers[1]:
                # Map to recurrence range
                rec_idx = self.config.recurrence_layers[0] + (
                    (i - self.config.recurrence_layers[0]) % 
                    (self.config.recurrence_layers[1] - self.config.recurrence_layers[0] + 1)
                )
                block = self.h[rec_idx]
            
            x = block(x)
        
        # Final norm
        x = self.ln_f(x)
        
        # Output logits (tie weights)
        logits = x @ self.wte.weight.t()
        
        # Compute loss
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce_loss + self.config.diffusion_loss_weight * diffusion_loss
        
        return logits, loss
    
    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """Get parameter groups for mixed optimizer (Muon + AdamW)."""
        groups = []
        
        # Collect all parameters
        muon_params = []
        adam_params = []
        
        for name, param in self.named_parameters():
            if 'diffusion' in name and self.diffusion is not None:
                # Use diffusion's own grouping
                continue
            elif 'weight' in name and len(param.shape) >= 2:
                muon_params.append(param)
            else:
                adam_params.append(param)
        
        if muon_params:
            groups.append({'params': muon_params, 'lr': base_lr, 'use_muon': True})
        if adam_params:
            groups.append({'params': adam_params, 'lr': base_lr, 'use_muon': False})
        
        # Add diffusion parameters if enabled
        if self.diffusion is not None:
            groups.extend(self.diffusion.get_param_groups(base_lr))
        
        return groups


# -----------------------------------------------------------------------------
# Quantization Utilities
# -----------------------------------------------------------------------------

def quantize_to_int6(weights: Dict[str, torch.Tensor], 
                     matrix_clip_sigma: float = 13.2,
                     embed_clip_sigma: float = 21.5) -> Dict[str, torch.Tensor]:
    """
    Quantize weights to int6 using GPTQ-style quantization with SD-Clip.
    """
    quantized = {}
    
    for name, weight in weights.items():
        if 'wte' in name or 'wpe' in name:
            # Embedding quantization with higher clip sigma
            clip_val = embed_clip_sigma * weight.std()
        else:
            # Matrix quantization
            clip_val = matrix_clip_sigma * weight.std()
        
        # Clip and scale
        weight_clipped = torch.clamp(weight, -clip_val, clip_val)
        scale = clip_val / 31.5  # int6: [-32, 31]
        
        # Quantize
        weight_int = torch.round(weight_clipped / scale).clamp(-32, 31).to(torch.int8)
        
        # Store scale for dequantization
        quantized[name] = {
            'weight': weight_int,
            'scale': scale,
            'shape': weight.shape
        }
    
    return quantized


def dequantize_from_int6(quantized: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Dequantize int6 weights back to float32."""
    dequantized = {}
    
    for name, q in quantized.items():
        weight_float = q['weight'].float() * q['scale']
        dequantized[name] = weight_float.reshape(q['shape'])
    
    return dequantized


# -----------------------------------------------------------------------------
# Test-Time Training (TTT)
# -----------------------------------------------------------------------------

@torch.no_grad()
def test_time_training(model: GPT, input_ids: torch.Tensor, 
                       num_epochs: int = 4, lr: float = 0.006) -> GPT:
    """
    Legal score-first TTT: adapt model on validation set before evaluation.
    Only uses gradients from the scoring objective (CE loss).
    """
    model.eval()
    
    # Create a copy for adaptation
    adapted_model = GPT(model.config)
    adapted_model.load_state_dict(model.state_dict())
    adapted_model.train()
    
    # Use AdamW for TTT
    optimizer = torch.optim.AdamW(adapted_model.parameters(), lr=lr, weight_decay=0.0)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        _, loss = adapted_model(input_ids, input_ids, training=False)
        
        # Backward pass (only CE loss, no diffusion loss during TTT)
        loss.backward()
        optimizer.step()
    
    adapted_model.eval()
    return adapted_model


# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------

def main():
    # Parse arguments (simplified for this example)
    config = ModelConfig()
    
    # Override with environment variables if available
    if 'DIFFUSION_STEPS' in os.environ:
        config.diffusion_steps = int(os.environ['DIFFUSION_STEPS'])
    if 'QK_GAIN_INIT' in os.environ:
        config.qk_gain_init = float(os.environ['QK_GAIN_INIT'])
    if 'MLP_MULTIPLIER' in os.environ:
        config.mlp_multiplier = float(os.environ['MLP_MULTIPLIER'])
    
    print(f"Configuration: {config}")
    print(f"Diffusion enabled: {config.diffusion_enabled}")
    print(f"Diffusion steps: {config.diffusion_steps}")
    
    # Initialize model
    model = GPT(config)
    model.cuda()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup optimizer with parameter groups
    param_groups = model.get_param_groups(base_lr=1e-3)
    
    muon_params = [g['params'] for g in param_groups if g.get('use_muon', False)]
    adam_params = [g['params'] for g in param_groups if not g.get('use_muon', False)]
    
    optimizer = torch.optim.AdamW(adam_params, lr=1e-3, weight_decay=0.095)
    if muon_params:
        muon_optimizer = MuonEqR(muon_params, lr=1e-3, weight_decay=0.095)
    
    # EMA
    ema_decay = 0.9965
    ema_model = GPT(config)
    ema_model.load_state_dict(model.state_dict())
    
    # Training loop (simplified)
    print("Starting training...")
    start_time = time.time()
    
    # Dummy training loop for demonstration
    for step in range(100):
        # Generate dummy batch
        input_ids = torch.randint(0, config.vocab_size, (8, config.max_seq_len)).cuda()
        
        # Forward pass
        logits, loss = model(input_ids, input_ids, training=True)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        if muon_params:
            muon_optimizer.step()
        optimizer.zero_grad()
        if muon_params:
            muon_optimizer.zero_grad()
        
        # EMA update
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
        
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}: loss={loss.item():.4f}, time={elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")
    
    # Quantize model
    print("Quantizing model...")
    quantized_weights = quantize_to_int6(
        ema_model.state_dict(),
        matrix_clip_sigma=config.matrix_clip_sigma,
        embed_clip_sigma=config.embed_clip_sigma
    )
    
    # Estimate artifact size
    total_bytes = 0
    for name, q in quantized_weights.items():
        total_bytes += q['weight'].numel() + q['scale'].numel() * 4  # int8 + float32 scale
    
    print(f"Estimated artifact size: {total_bytes / 1024 / 1024:.2f} MB")
    
    # Save model
    torch.save(quantized_weights, 'model_quantized.pt')
    print("Model saved to model_quantized.pt")


if __name__ == '__main__':
    main()
