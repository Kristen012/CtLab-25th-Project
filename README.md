# CtLab-25th-Project

FPGA implementation of a GPT-2 style transformer inference pipeline using Xilinx Vitis/XRT. The top-level host (`host.cpp`) tokenizes input text, applies word/position embeddings, runs 12 transformer blocks (layer norm → attention with KV cache → residual add → MLP → residual add), projects logits to the vocabulary, and performs top-k sampling for autoregressive generation.

## Repository Layout
- `host.cpp`, `host.hpp` – Top-level OpenCL host: device discovery, buffer management, weight loading, per-iteration execution loop, and sampling.
- `attention/` – Attention kernels (`kernel_attn_tiling.cpp`, `kernel_attn_proj.cpp`) plus a standalone host for kernel testing.
- `layernorm/` – Layer norm kernel and host harness.
- `Vadd/` – Vector add/residual add kernel and host harness.
- `MLP/` – Feed-forward (c_fc + c_proj) kernel and host harness.
- `last_linear/` – Final vocabulary projection kernels.
- `WPE_WTE_add/` – Word and positional embedding helpers (sum WTE/WPE for a sequence).
- `tokenization/` – GPT-2 BPE vocab/merges and a tokenizer implementation.
- `sample/` – Top-k sampling utilities and sample token outputs.
- `decode/` – Detokenization helpers for converting generated ids back to text.
- `dequant/` – Experimental int4 → fp32 dequantization kernels for MLP weights.
