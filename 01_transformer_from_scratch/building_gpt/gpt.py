import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
from datetime import datetime

# Hyperparameters
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2

# Set seed for reproducibility
torch.manual_seed(135)

def load_and_preprocess_data(file_path='input.txt'):
    """Load and preprocess the text data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded text with {len(text)} characters")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {file_path}. Please download the dataset first.")
    
    # Create character mappings
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(s): return [stoi[c] for c in s]
    def decode(l): return ''.join([itos[i] for i in l])
    
    return text, chars, vocab_size, stoi, itos, encode, decode

def create_train_val_splits(data, train_ratio=0.9):
    """Create training and validation splits."""
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(split, train_data, val_data, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE):
    """Generate a batch of data for training or validation."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Estimate loss on training and validation sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """One head of self-attention."""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # Scaled dot-product attention
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Weighted aggregation of values
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Feed-forward neural network."""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # Using GELU instead of ReLU for better performance
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation."""
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Residual connection + layer norm
        x = x + self.ffwd(self.ln2(x))  # Residual connection + layer norm
        return x

class GPTLanguageModel(nn.Module):
    """GPT Language Model implementation."""
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        
        self.blocks = nn.Sequential(*[
            Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)
        ])
        
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        
        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens given a context."""
        for _ in range(max_new_tokens):
            # Crop context to block size
            idx_cond = idx[:, -BLOCK_SIZE:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus on last time step
            logits = logits[:, -1, :]
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

def save_model(model, optimizer, vocab_size, stoi, itos, filepath='gpt_model.pth'):
    """Save the trained model and related information."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'block_size': BLOCK_SIZE,
            'n_embd': N_EMBD,
            'n_head': N_HEAD,
            'n_layer': N_LAYER,
            'dropout': DROPOUT
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath='gpt_model.pth'):
    """Load a trained model and related information."""
    checkpoint = torch.load(filepath, map_location=DEVICE)
    
    model = GPTLanguageModel(checkpoint['vocab_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint['stoi'], checkpoint['itos']

def main():
    """Main training function."""
    print(f"Using device: {DEVICE}")
    
    # Load and preprocess data
    text, chars, vocab_size, stoi, itos, encode, decode = load_and_preprocess_data()
    data = torch.tensor(encode(text), dtype=torch.long)
    train_data, val_data = create_train_val_splits(data)
    
    # Create model
    model = GPTLanguageModel(vocab_size)
    model = model.to(DEVICE)
    
    # Print model information
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model has {num_params:.2f}M parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("Starting training...")
    start_time = datetime.now()
    
    for iter in range(MAX_ITERS):
        # Evaluate loss periodically
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            elapsed_time = datetime.now() - start_time
            print(f"Step {iter:5d} | Train loss: {losses['train']:.4f} | "
                  f"Val loss: {losses['val']:.4f} | Time: {elapsed_time}")
        
        # Training step
        xb, yb = get_batch('train', train_data, val_data)
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Save the trained model
    save_model(model, optimizer, vocab_size, stoi, itos)
    
    # Generate sample text
    print("\nGenerating sample text...")
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated_tokens = model.generate(context, max_new_tokens=500)
    generated_text = decode(generated_tokens[0].tolist())
    print(f"\nGenerated text:\n{generated_text}")
    
    # Save generated text to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_text_{timestamp}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(generated_text)
    print(f"\nGenerated text saved to {filename}")

if __name__ == "__main__":
    main()