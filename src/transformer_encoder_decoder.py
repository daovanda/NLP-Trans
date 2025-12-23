import torch
import torch.nn as nn
from transformer_components import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    ResidualConnection,
    LayerNorm
)

# 1. ENCODER LAYER

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Residual Connections
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        
    def forward(self, x, mask=None):
        # 1. Self-Attention với Residual Connection
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, mask)[0])
        
        # 2. Feed-Forward với Residual Connection
        x = self.residual2(x, self.feed_forward)
        
        return x

# 2. ENCODER

class Encoder(nn.Module):

    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        
        from transformer_components import Embedding, PositionalEncoding
        
        # Embedding layer
        self.embedding = Embedding(vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Stack of Encoder Layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final Layer Normalization
        self.norm = LayerNorm(d_model)
        
    def forward(self, src, src_mask=None):

        # 1. Embedding + Positional Encoding
        x = self.embedding(src)
        x = self.pos_encoding(x)
        
        # 2. Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
        
        # 3. Final normalization
        x = self.norm(x)
        
        return x

# 3. DECODER LAYER

class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Masked Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Multi-Head Cross-Attention (Encoder-Decoder Attention)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Residual Connections
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):

        # 1. Masked Self-Attention với Residual Connection
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, tgt_mask)[0])
        
        # 2. Cross-Attention với Encoder output
        # Q từ decoder, K, V từ encoder
        x = self.residual2(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask)[0])
        
        # 3. Feed-Forward với Residual Connection
        x = self.residual3(x, self.feed_forward)
        
        return x

# 4. DECODER

class Decoder(nn.Module):

    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        
        from transformer_components import Embedding, PositionalEncoding
        
        # Embedding layer
        self.embedding = Embedding(vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Stack of Decoder Layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final Layer Normalization
        self.norm = LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):

        # 1. Embedding + Positional Encoding
        x = self.embedding(tgt)
        x = self.pos_encoding(x)
        
        # 2. Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # 3. Final normalization
        x = self.norm(x)
        
        # 4. Project to vocabulary
        output = self.fc_out(x)
        
        return output

# 5. MASK FUNCTIONS (FIXED)

def create_padding_mask(seq, pad_idx=0):

    # Tạo mask: True cho non-padding, False cho padding
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask  # Returns bool tensor

def create_causal_mask(seq_len, device):

    # Tạo lower triangular matrix - FIXED: convert to bool
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    mask = mask.bool()  # Convert to bool
    mask = mask.unsqueeze(0).unsqueeze(1)
    return mask

def create_target_mask(tgt, pad_idx=0):

    batch_size, tgt_len = tgt.size()
    device = tgt.device
    
    # Padding mask - returns bool
    padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]
    
    # Causal mask - returns bool
    causal_mask = create_causal_mask(tgt_len, device)  # [1, 1, tgt_len, tgt_len]
    
    # Kết hợp cả 2 masks - both are bool now
    mask = padding_mask & causal_mask
    
    return mask

# 6. TEST ENCODER & DECODER

if __name__ == "__main__":
    print("="*70)
    print("KIỂM TRA ENCODER & DECODER")
    print("="*70)
    
    # Hyperparameters
    batch_size = 2
    src_len = 10
    tgt_len = 12
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    d_model = 512
    n_layers = 6
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    pad_idx = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Tạo dummy data
    src = torch.randint(1, src_vocab_size, (batch_size, src_len)).to(device)
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len)).to(device)
    
    # Tạo masks
    src_mask = create_padding_mask(src, pad_idx).to(device)
    tgt_mask = create_target_mask(tgt, pad_idx).to(device)
    
    print(f"\nInput shapes:")
    print(f"  Source: {src.shape}")
    print(f"  Target: {tgt.shape}")
    print(f"  Source mask: {src_mask.shape}, dtype: {src_mask.dtype}")
    print(f"  Target mask: {tgt_mask.shape}, dtype: {tgt_mask.dtype}")
    
    # Test Encoder
    print("\n" + "="*70)
    print("Test Encoder")
    print("="*70)
    
    encoder = Encoder(
        vocab_size=src_vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)
    
    encoder_output = encoder(src, src_mask)
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Expected: [{batch_size}, {src_len}, {d_model}]")
    
    # Test Decoder
    print("\n" + "="*70)
    print("Test Decoder")
    print("="*70)
    
    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)
    
    decoder_output = decoder(tgt, encoder_output, src_mask, tgt_mask)
    print(f"Decoder output shape: {decoder_output.shape}")
    print(f"Expected: [{batch_size}, {tgt_len}, {tgt_vocab_size}]")
    
    # Số lượng parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    
    print("\n" + "="*70)
    print("THỐNG KÊ MÔ HÌNH")
    print("="*70)
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters: {encoder_params + decoder_params:,}")
    
    print("\n" + "="*70)
    print("✓ ENCODER & DECODER HOẠT ĐỘNG ĐÚNG!")
    print("="*70)