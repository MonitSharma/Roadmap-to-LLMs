import torch 
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

# constructor of the input embeddings

class InputEmbeddings(nn.Module):

    def __init__(self, vocab_size, d_model):   # d_model : dimension of the model
                                               # vocab_size : size of the vocabulary
        super(InputEmbeddings, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x.long()) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))


    # this vector is learned by the model



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_length, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Create the positional encodings
        encoding = torch.zeros(seq_length, d_model)
        for pos in range(seq_length):
            for i in range(0, d_model, 2):
                angle = pos / (10000 ** (i / d_model))
                encoding[pos, i] = torch.sin(torch.tensor(angle, dtype=torch.float32))
                if i + 1 < d_model:
                    encoding[pos, i + 1] = torch.cos(torch.tensor(angle, dtype=torch.float32))
        self.register_buffer('encoding', encoding.unsqueeze(0))  
        # This ensures encoding is moved with the model to the correct device and is included in model state dicts, but is not a learnable parameter.

    def forward(self, x):
        x = x + self.encoding[:, :x.size(1)]

        # By registering encoding as a buffer (with register_buffer), it is already excluded from gradient computation and will not be updated during training.
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_model))  # Scale parameter alpha multiplied
        self.b_2 = nn.Parameter(torch.zeros(d_model)) # Shift parameter beta added

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout): # from the paper
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # W1 and b1
        self.linear2 = nn.Linear(d_ff, d_model)  # W2 and b2 
        self.dropout = nn.Dropout(dropout)       

    def forward(self, x):
        x = self.linear1(x)        
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out    = nn.Linear(d_model, d_model)
        self.drop   = nn.Dropout(dropout)

    @staticmethod
    def _mask_scores(scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # scores: [B,H,T,S]; mask: bool keep-mask, e.g. [B,1,T,S] or [B,1,1,S]
        if mask is None:
            return scores
        if mask.dtype != torch.bool:
            mask = mask != 0
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)               # [B,1,T,S]
        # Broadcast to [B,H,T,S]
        return scores.masked_fill(~mask, float("-inf"))

    def forward(self, query, key, value, mask=None, return_attn_weights=False):
        # Strictly require [B, T, d_model] at the API boundary to avoid re-splitting
        assert query.dim() == 3 and key.dim() == 3 and value.dim() == 3, \
            f"Expected [B,T,d_model], got {query.shape}, {key.shape}, {value.shape}"

        B, Tq, _ = query.shape
        Bk, Tk, _ = key.shape
        assert B == Bk == value.size(0) and Tk == value.size(1)

        # Project then split once: [B,T,d_model] -> [B,H,T,d_k]
        Q = self.q_proj(query).view(B, Tq, self.num_heads, self.d_k).permute(0, 2, 1, 3).contiguous()
        K = self.k_proj(key  ).view(B, Tk, self.num_heads, self.d_k).permute(0, 2, 1, 3).contiguous()
        V = self.v_proj(value).view(B, Tk, self.num_heads, self.d_k).permute(0, 2, 1, 3).contiguous()

        assert Q.shape == (B, self.num_heads, Tq, self.d_k), Q.shape
        assert K.shape == (B, self.num_heads, Tk, self.d_k), K.shape
        assert V.shape == (B, self.num_heads, Tk, self.d_k), V.shape


        # Dot-product without accidental broadcasting: [B,H,Tq,d_k] x [B,H,Tk,d_k] -> [B,H,Tq,Tk]
        scores = torch.einsum("bhtd,bhsd->bhts", Q, K) / (self.d_k ** 0.5)
        scores = self._mask_scores(scores, mask)

        assert scores.dim() == 4 and scores.shape[:2] == (B, self.num_heads), scores.shape


        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        # Context: [B,H,Tq,Tk] x [B,H,Tk,d_k] -> [B,H,Tq,d_k]
        ctx = torch.einsum("bhts,bhsd->bhtd", attn, V)

        # Merge heads: [B,H,Tq,d_k] -> [B,Tq,d_model]
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, Tq, self.d_model)
        out = self.out(ctx)

        return (out, attn) if return_attn_weights else out


class ResidualConnection(nn.Module):

    def __init__(self, d_model, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x, lambda x: self.attention(x, x, x, mask))
        x = self.residual2(x, self.feed_forward)
        return x
    
class Encoder(nn.Module):

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    


class DecoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, return_attn_weights=False):
        attn_weights = {}
        def self_attn_fn(y):
            if return_attn_weights:
                out, attn = self.self_attention(y, y, y, tgt_mask, True)
                attn_weights['self'] = attn
                return out
            return self.self_attention(y, y, y, tgt_mask)
        def cross_attn_fn(y):
            if return_attn_weights:
                out, attn = self.cross_attention(y, enc_output, enc_output, src_mask, True)
                attn_weights['cross'] = attn
                return out
            return self.cross_attention(y, enc_output, enc_output, src_mask)
        x = self.residual1(x, self_attn_fn)
        x = self.residual2(x, cross_attn_fn)
        x = self.residual3(x, self.feed_forward)
        if return_attn_weights:
            return x, attn_weights
        return x
    
class Decoder(nn.Module):

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = LayerNormalization(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, return_attn_weights=False):
        attn_weights_all = [] if return_attn_weights else None
        for layer in self.layers:
            if return_attn_weights:
                x, attn_weights = layer(x, enc_output, src_mask, tgt_mask, True)
                attn_weights_all.append(attn_weights)
            else:
                x = layer(x, enc_output, src_mask, tgt_mask)
        x = self.norm(x)
        if return_attn_weights:
            return x, attn_weights_all
        return x
    

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # return F.log_softmax(self.linear(x), dim=-1)
        return self.linear(x)
    

class Transformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1, max_seq_length=100):
        super(Transformer, self).__init__()
        self.src_embedding = InputEmbeddings(src_vocab_size, d_model)
        self.tgt_embedding = InputEmbeddings(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        self.projection = ProjectionLayer(d_model, tgt_vocab_size)
        self.init_weights()

    def init_weights(self):
        for p in self.modules():
            if isinstance(p, nn.Linear):
                nn.init.xavier_uniform_(p.weight)
                if p.bias is not None:
                    nn.init.zeros_(p.bias)
            elif isinstance(p, nn.Embedding):
                nn.init.xavier_uniform_(p.weight)

    def encode(self, src, src_mask=None):
        src_emb = self.positional_encoding(self.src_embedding(src))
        return self.encoder(src_emb, src_mask)

    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None, return_attn_weights=False):
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))
        return self.decoder(tgt_emb, enc_output, src_mask, tgt_mask, return_attn_weights=return_attn_weights)

    def project(self, dec_output):
        return self.projection(dec_output)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, return_attn_weights=False):
        enc_output = self.encode(src, src_mask)
        if return_attn_weights:
            dec_output, attn_weights = self.decode(tgt, enc_output, src_mask, tgt_mask, True)
            output = self.project(dec_output)
            return output, attn_weights
        else:
            dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
            output = self.project(dec_output)
            return output

def build_transformer(src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1, max_seq_length=100):
    return Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, dropout, max_seq_length)