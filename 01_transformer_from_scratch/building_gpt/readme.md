# GPT Language Model Implementation

This repository contains a full implementation of a GPT-style transformer language model trained on the Tiny Shakespeare dataset. The model learns to generate text character-by-character through self-attention mechanisms.

## 📋 Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Mathematical Explanation](#mathematical-explanation)
- [Setup](#setup)
- [Usage](#usage)
- [Training Details](#training-details)
- [Results](#results)

## 🌟 Features

- Full transformer architecture with multi-head self-attention
- Character-level tokenization
- Causal language modeling (next character prediction)
- GPU acceleration support
- Layer normalization and residual connections
- Positional embeddings for sequence order awareness

## 🏗️ Architecture

The model implements a decoder-only transformer with the following components:

- **6 Transformer Blocks**
- **6 Multi-Head Attention Layers** (6 heads each)
- **Feed-Forward Networks** with ReLU activation
- **Layer Normalization** and residual connections
- **Embedding Dimension**: 384
- **Context Length**: 256 characters
- **Vocabulary Size**: 65 unique characters

## 🧮 Mathematical Explanation

### 1. Input Embedding and Positional Encoding

Input tokenization and embedding:

$$
\text{Input: } \mathbf{idx} \in \mathbb{R}^{B \times T} \text{ where } B = \text{batch\_size}, T = \text{block\_size}
$$

$$
\text{Token Embedding: } \mathbf{E}_{\text{token}} \in \mathbb{R}^{\text{vocab\_size} \times \text{n\_embd}}
$$

$$
\text{Position Embedding: } \mathbf{E}_{\text{pos}} \in \mathbb{R}^{\text{block\_size} \times \text{n\_embd}}
$$

$$\text{Output: } \mathbf{X} = \mathbf{E}_{token}[\mathbf{idx}] + \mathbf{E}_{pos}[0:T] \in \mathbb{R}^{B \times T \times n\_embd}$$

**Example:**
$$\mathbf{idx} = \begin{bmatrix} [1, 5, 3, 2] \\ [7, 2, 8, 1] \end{bmatrix} \text{ (First and second sequences)}$$

$$\text{After combining token + position embeddings: } \mathbf{X} \in \mathbb{R}^{2 \times 4 \times 384}$$

### 2. Self-Attention Head

For each attention head:
$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q \text{ where } \mathbf{W}_Q \in \mathbb{R}^{n\_embd \times head\_size}$$
$$\mathbf{K} = \mathbf{X}\mathbf{W}_K \text{ where } \mathbf{W}_K \in \mathbb{R}^{n\_embd \times head\_size}$$
$$\mathbf{V} = \mathbf{X}\mathbf{W}_V \text{ where } \mathbf{W}_V \in \mathbb{R}^{n\_embd \times head\_size}$$

Attention computation:
$$\text{Attention Scores: } \mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{head\_size}} \in \mathbb{R}^{B \times T \times T}$$
$$\text{Masked Softmax: } \mathbf{A} = \text{softmax}(\text{mask}(\mathbf{S})) \in \mathbb{R}^{B \times T \times T}$$
$$\text{Output: } \mathbf{O} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{B \times T \times head\_size}$$

**Attention Mechanism Example:**
$$\text{Compute attention scores: } \mathbf{S} = \mathbf{Q} \mathbf{K}^T / \sqrt{head\_size}$$

$$\text{Apply causal mask (upper triangle = } -\infty\text{):}$$
$$\mathbf{S}_{masked} = \begin{bmatrix}
1.56 & -\infty & -\infty & -\infty \\
1.70 & 2.30 & -\infty & -\infty \\
1.20 & 1.90 & 1.10 & -\infty \\
1.40 & 2.10 & 1.20 & 1.90
\end{bmatrix}$$

$$\text{Apply softmax:}$$
$$\mathbf{A} = \begin{bmatrix}
1.00 & 0.00 & 0.00 & 0.00 \\
0.35 & 0.65 & 0.00 & 0.00 \\
0.30 & 0.40 & 0.30 & 0.00 \\
0.25 & 0.35 & 0.20 & 0.20
\end{bmatrix}$$

### 3. Multi-Head Attention

$$\text{MultiHead}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)\mathbf{W}_O$$

Where:
- $\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_Q^i, \mathbf{K}\mathbf{W}_K^i, \mathbf{V}\mathbf{W}_V^i)$
- $\mathbf{W}_O \in \mathbb{R}^{h \times head\_size \times n\_embd}$

### 4. Feed-Forward Network

$$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

Where:
- $\mathbf{W}_1 \in \mathbb{R}^{n\_embd \times 4 \times n\_embd}$ 
- $\mathbf{W}_2 \in \mathbb{R}^{4 \times n\_embd \times n\_embd}$

### 5. Transformer Block

First sub-layer (Multi-head attention with residual connection):
$$\mathbf{z}_1 = \text{MultiHeadAttention}(\text{LayerNorm}(\mathbf{x}))$$
$$\mathbf{x} = \mathbf{x} + \mathbf{z}_1$$

Second sub-layer (Feed-forward with residual connection):
$$\mathbf{z}_2 = \text{FeedForward}(\text{LayerNorm}(\mathbf{x}))$$
$$\mathbf{x} = \mathbf{x} + \mathbf{z}_2$$

### 6. Full Forward Pass

$$\text{Input: } \mathbf{idx} \in \mathbb{R}^{B \times T}$$
$$\downarrow \text{Token + Position Embedding}$$
$$\mathbf{X}^{(0)} \in \mathbb{R}^{B \times T \times 384}$$
$$\downarrow \text{Block 1: Attention + FFN + Residuals}$$
$$\mathbf{X}^{(1)} \in \mathbb{R}^{B \times T \times 384}$$
$$\downarrow \text{Block 2: Attention + FFN + Residuals}$$
$$\mathbf{X}^{(2)} \in \mathbb{R}^{B \times T \times 384}$$
$$\vdots$$
$$\downarrow \text{Block 6: Attention + FFN + Residuals}$$
$$\mathbf{X}^{(6)} \in \mathbb{R}^{B \times T \times 384}$$
$$\downarrow \text{Final Layer Norm}$$
$$\mathbf{X}_{final} \in \mathbb{R}^{B \times T \times 384}$$
$$\downarrow \text{Linear Projection}$$
$$\text{logits} \in \mathbb{R}^{B \times T \times vocab\_size}$$

### 7. Loss Computation

$$\text{Loss} = \text{CrossEntropy}(\text{logits}, \text{targets})$$

### 8. Generation Process

The generation process follows these steps:
1. Start with context: $\mathbf{idx} \in \mathbb{R}^{1 \times 1}$
2. For each new token:
    1. Take last block_size tokens: $\mathbf{idx}_{cond} = \mathbf{idx}[:, -block\_size:]$
    2. Forward pass: $\text{logits}, \_ = \text{model}(\mathbf{idx}_{cond})$
    3. Focus on last position: $\text{logits} = \text{logits}[:, -1, :] \in \mathbb{R}^{1 \times vocab\_size}$
    4. Apply softmax: $\text{probs} = \text{softmax}(\text{logits}) \in \mathbb{R}^{1 \times vocab\_size}$
    5. Sample next token: $\mathbf{idx}_{next} \sim \text{Multinomial}(\text{probs})$
    6. Append: $\mathbf{idx} = \text{concat}(\mathbf{idx}, \mathbf{idx}_{next})$

## 🚀 Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd <repository-name>
```

2. **Install dependencies:**
```bash
uv pip install torch
```

3. **Download the Dataset:**
The code automatically downloads the Tiny Shakespeare dataset, or you can manually download it:
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## ▶️ Usage

Run the training script

```bash
python gpt.py
```



### **References**

1. [Let's build GPT by Andrej Karpathy](https://youtu.be/kCc8FmEb1nY?si=2FVmAeudSOD5d0W2)
2. [nanoGPT](https://github.com/karpathy/nanoGPT)
3. [Attention is all you need](https://github.com/MonitSharma/Roadmap-to-LLMs/tree/main/01_transformer_from_scratch/notes)