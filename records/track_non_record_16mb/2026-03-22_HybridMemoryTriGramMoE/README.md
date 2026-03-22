# Hybrid Retentive Memory + TriGram Hash + Routed SwiGLU

## Status

This is a **non-record exploratory submission** aimed at pushing the parameter-golf baseline toward a more modern hybrid LLM design. The goal is not to claim a leaderboard result yet, but to package a runnable training script that combines several recent ideas in a 16MB-oriented setting.

## Main ideas

Starting from the recent strong SmearGate + BigramHash recipe, this variant adds four extra ingredients:

1. **TriGram hash embeddings**
   - Extends the hashed n-gram idea from token pairs to token triplets.
   - Uses a compact 8192-bucket learned table with a small projection back to model width.
   - Intended to cheaply capture short phrase patterns that a tiny model often struggles to represent.

2. **Retentive feature memory**
   - Adds a lightweight recurrent memory path inspired by linear-attention / retention-style models.
   - Each block maintains an exponentially decayed feature state over the sequence and mixes it back into the residual stream.
   - This is meant to improve long-range signal propagation without paying for extra full attention heads.

3. **Causal depthwise convolution branch**
   - A tiny local sequence mixer that complements attention with cheap short-range pattern extraction.
   - Useful in small models where attention can be parameter-starved.

4. **Routed SwiGLU MLP**
   - Replaces the plain squared-ReLU MLP with a small soft-routed two-expert SwiGLU-style feedforward block.
   - This gives a mild mixture-of-experts flavor while keeping all experts active and implementation-simple.

## Expected tradeoffs

- **Potential upside:** better utilization of limited parameters by mixing token-level, n-gram, local convolutional, recurrent-memory, and attention mechanisms.
- **Potential downside:** the recurrent memory loop may cost some training throughput; the extra branches also make quantization behavior less obvious than the simpler leaderboard models.

## Suggested first sweep

```bash
RUN_ID=hybrid_memory_trigram \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 \
NUM_LAYERS=10 \
MLP_MULT=2.75 \
TRIGRAM_VOCAB_SIZE=8192 \
TRIGRAM_DIM=96 \
MEMORY_DIM=128 \
EXPERTS=2 \
WEIGHT_DECAY=0.03 \
python train_gpt.py
```

## Notes

- This run folder is intended as a starting point for future tuning, not a verified SOTA claim.
- I have only smoke-checked script compilation in this environment; no 8xH100 benchmark result is claimed here.
