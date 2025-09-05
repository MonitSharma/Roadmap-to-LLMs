import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

torch.manual_seed(135)
# Download dataset if not present
if not os.path.exists("input.txt"):
    print("Downloading Shakespeare dataset...")
    os.system("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

# Read dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Length of dataset in characters: {len(text):,}")
print("\nFirst 1000 characters:")
print(text[:1000])

# Character-level tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"\nUnique characters ({vocab_size}): {''.join(chars)}")

# Mapping from character to index and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encoder and decoder functions
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

# Test encoding/decoding
test_str = "hii there"
encoded = encode(test_str)
decoded = decode(encoded)
print(f"\nEncoded '{test_str}' -> {encoded}")
print(f"Decoded {encoded} -> '{decoded}'")

# Convert full text to tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(f"\nData shape: {data.shape}, dtype: {data.dtype}")
print("First 1000 tokens:", data[:1000])

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"\nTrain data size: {train_data.shape}")
print(f"Val data size: {val_data.shape}")

# Batch preparation function
def get_batch(split, batch_size=4, block_size=8):
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - block_size, (batch_size,))
    x = torch.stack([data_[i:i+block_size] for i in ix])
    y = torch.stack([data_[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Demo batch
xb, yb = get_batch('train')
print("\nSample input batch:")
print(xb)
print("Sample target batch:")
print(yb)

# Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # Take last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Initialize model and move to device
model = BigramLanguageModel(vocab_size).to(device)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())} total")

# Sample generation before training
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\nBefore training sample generation:")
print(decode(model.generate(idx, max_new_tokens=100)[0].tolist()))

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
batch_size = 32
block_size = 8
eval_interval = 500
eval_iters = 200

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training
for step in range(10000):
    # Get batch
    xb, yb = get_batch('train', batch_size, block_size)

    # Forward pass
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Print progress
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {step:>6} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

# Final sample generation
print("\nAfter training sample generation:")
generated = model.generate(idx, max_new_tokens=500)[0].tolist()
print(decode(generated))