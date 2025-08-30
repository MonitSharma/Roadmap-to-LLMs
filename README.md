# 🚀 LLM Roadmap: From Scratch to State-of-the-Art

> **Build, Train, and Deploy Large Language Models — with code in Python, PyTorch, CUDA, and Apple MLX**

This repository is a **hands-on, code-first journey** into the world of Large Language Models (LLMs). Whether you're a student, researcher, or engineer, this roadmap guides you from implementing a **transformer from scratch** to exploring cutting-edge architectures like **Mamba**, **MoE**, and **FlashAttention** — all with runnable code.

🎯 **Goal**: Understand LLMs not just as black boxes, but as systems you can **build, optimize, and deploy**.

🔧 **Tech Stack**: Python, PyTorch, CUDA, Triton, C++, MLX (Apple), GGUF, ONNX

---

## 📚 Table of Contents

1. [Overview](#-overview)
2. [Roadmap](#-roadmap)
3. [Features](#-features)
4. [Getting Started](#-getting-started)
5. [Folder Structure](#-folder-structure)
6. [Requirements](#-requirements)
7. [Usage](#-usage)
8. [Contributing](#-contributing)
9. [License](#-license)
10. [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

This repo breaks down the **full stack of LLM development** into digestible, code-driven lessons:

- ✅ **From Scratch**: Implement attention, embeddings, and training loops.
- ⚙️ **Systems**: Distributed training, quantization, and inference optimization.
- 🚀 **Cutting-Edge**: MoE, RLHF, FlashAttention, Mamba, and KANs.
- 💻 **Multi-Backend**: Code examples in **PyTorch**, **CUDA**, and **Apple MLX**.

Each chapter includes:
- Clear explanations
- Minimal, readable code
- Benchmarks and performance tips
- References to papers and resources

---

## 🗺️ Roadmap

| Chapter | Title | Key Topics | Code Focus |
|--------|------|-----------|-----------|
| 01 | [Transformer from Scratch](01_transformer_from_scratch/) | Attention, BPE, Training Loop | NumPy, PyTorch |
| 02 | [Distributed Training](02_distributed_training/) | DDP, FSDP, Pipeline Parallel | PyTorch, `torch.distributed` |
| 03 | [RLHF](03_rlhf/) | Reward Modeling, PPO, KL Control | TRL, Custom PPO |
| 04 | [Efficient Attention](04_efficient_attention/) | FlashAttention, Sparse Attn | CUDA, Triton |
| 05 | [Emerging Architectures](05_emerging_architectures/) | MoE, Mamba, KANs | PyTorch, SSMs |
| 06 | [Quantization & Sparsity](06_quantization/) | GGUF, GPTQ, Pruning | `llama.cpp`, `bitsandbytes` |
| 07 | [Inference Engineering](07_inference_engineering/) | KV Cache, Speculative Decoding | Python, MLX |
| 08 | [Systems & Deployment](08_deployment/) | ONNX, Triton, FastAPI | Docker, REST API |
| 09 | [Advanced Topics](09_advanced_topics/) | PAL, Neural ODEs, Self-Improvement | Research ideas |
| 10 | [Capstone Project](10_capstone/) | Build & Serve Your Mini-GPT | End-to-end pipeline |

---

## 🌟 Features

- ✅ **Code-first learning**: Every concept has a working implementation.
- 🍏 **Apple MLX support**: Run models efficiently on M1/M2/M3 Macs.
- ⚡ **CUDA kernels**: Explore optimized attention and linear layers.
- 📈 **Benchmarks**: Compare speed, memory, and accuracy across methods.
- 🧪 **Tiny datasets**: Train fast on `tiny-shakespeare` or synthetic data.
- 🐳 **Docker support**: Reproducible environments.
- 🤝 **Community-driven**: Contributions welcome!

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch (with CUDA support if available)
- Optional: CUDA toolkit, Triton, MLX, Docker

### Clone and Setup
```bash
uv pip install -r requirements.txt
```