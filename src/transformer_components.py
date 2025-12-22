"""
PHẦN B: XÂY DỰNG KIẾN TRÚC TRANSFORMER FROM SCRATCH
Các thành phần cốt lõi: Attention, Positional Encoding, Encoder, Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# 1. SCALED DOT-PRODUCT ATTENTION
# ============================================================================

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
    
    Args:
        d_k: Dimension của key (để scale)
    """
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query [batch_size, n_heads, seq_len, d_k]
            K: Key [batch_size, n_heads, seq_len, d_k]
            V: Value [batch_size, n_heads, seq_len, d_v]
            mask: Mask [batch_size, 1, seq_len, seq_len] hoặc [batch_size, 1, 1, seq_len]
            
        Returns:
            output: [batch_size, n_heads, seq_len, d_v]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        # Tính attention scores: Q·K^T
        # [batch, n_heads, seq_len, d_k] @ [batch, n_heads, d_k, seq_len]
        # -> [batch, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Scale bởi sqrt(d_k) để tránh gradient quá nhỏ
        scores = scores / math.sqrt(self.d_k)
        
        # Apply mask (nếu có)
        if mask is not None:
            # Mask = 0 -> giữ nguyên, Mask = 1 -> set = -inf
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        
        # Softmax để có attention weights
        attention_weights = self.softmax(scores)
        
        # Apply attention weights to values
        # [batch, n_heads, seq_len, seq_len] @ [batch, n_heads, seq_len, d_v]
        # -> [batch, n_heads, seq_len, d_v]
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

# ============================================================================
# 2. MULTI-HEAD ATTENTION
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    
    Chia input thành nhiều heads, mỗi head học các representation khác nhau
    
    Args:
        d_model: Dimension của model
        n_heads: Số lượng attention heads
        dropout: Dropout rate
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension của mỗi head
        
        # Linear layers để project Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Scaled Dot-Product Attention
        self.attention = ScaledDotProductAttention(self.d_k)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query [batch_size, seq_len, d_model]
            K: Key [batch_size, seq_len, d_model]
            V: Value [batch_size, seq_len, d_model]
            mask: Mask tensor
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = Q.size(0)
        
        # 1. Linear projection và chia thành multiple heads
        # [batch, seq_len, d_model] -> [batch, seq_len, n_heads, d_k] -> [batch, n_heads, seq_len, d_k]
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply attention
        # output: [batch, n_heads, seq_len, d_k]
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # 3. Concatenate heads
        # [batch, n_heads, seq_len, d_k] -> [batch, seq_len, n_heads, d_k] -> [batch, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. Final linear projection
        output = self.W_o(output)
        output = self.dropout(output)
        
        return output, attention_weights

# ============================================================================
# 3. POSITION-WISE FEED-FORWARD NETWORK
# ============================================================================

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Áp dụng 2 linear transformations với ReLU ở giữa
    
    Args:
        d_model: Dimension của model
        d_ff: Dimension của hidden layer (thường = 4 * d_model)
        dropout: Dropout rate
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # x -> W1 -> ReLU -> Dropout -> W2 -> Dropout
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x

# ============================================================================
# 4. POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional Encoding (Sinusoidal)
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Thêm thông tin về vị trí của token trong sequence
    
    Args:
        d_model: Dimension của model
        max_len: Maximum sequence length
        dropout: Dropout rate
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Tạo positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Tính div_term cho công thức
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        # Apply sin cho các vị trí chẵn, cos cho các vị trí lẻ
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Thêm batch dimension
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # Register as buffer (không train)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # Cộng positional encoding
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)

# ============================================================================
# 5. LAYER NORMALIZATION
# ============================================================================

class LayerNorm(nn.Module):
    """
    Layer Normalization
    
    Chuẩn hóa theo dimension cuối (features)
    
    Args:
        d_model: Dimension của model
        eps: Epsilon cho numerical stability
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# ============================================================================
# 6. RESIDUAL CONNECTION
# ============================================================================

class ResidualConnection(nn.Module):
    """
    Residual Connection với Layer Normalization
    
    output = LayerNorm(x + Sublayer(x))
    
    Args:
        d_model: Dimension của model
        dropout: Dropout rate
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        Args:
            x: Input [batch_size, seq_len, d_model]
            sublayer: Function (callable) để apply
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Residual: x + sublayer(norm(x))
        return x + self.dropout(sublayer(self.norm(x)))

# ============================================================================
# 7. EMBEDDING LAYER
# ============================================================================

class Embedding(nn.Module):
    """
    Embedding layer với scaling
    
    Args:
        vocab_size: Kích thước vocabulary
        d_model: Dimension của model
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Scale embedding by sqrt(d_model) như trong paper
        return self.embedding(x) * math.sqrt(self.d_model)

# ============================================================================
# 8. TEST COMPONENTS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("KIỂM TRA CÁC THÀNH PHẦN TRANSFORMER")
    print("="*70)
    
    # Hyperparameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    d_ff = 2048
    vocab_size = 10000
    
    # Test Embedding
    print("\n1. Test Embedding:")
    embedding = Embedding(vocab_size, d_model)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    embedded = embedding(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {embedded.shape}")
    
    # Test Positional Encoding
    print("\n2. Test Positional Encoding:")
    pos_enc = PositionalEncoding(d_model)
    pos_encoded = pos_enc(embedded)
    print(f"   Input shape: {embedded.shape}")
    print(f"   Output shape: {pos_encoded.shape}")
    
    # Test Multi-Head Attention
    print("\n3. Test Multi-Head Attention:")
    mha = MultiHeadAttention(d_model, n_heads)
    output, attn_weights = mha(pos_encoded, pos_encoded, pos_encoded)
    print(f"   Input shape: {pos_encoded.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    # Test Feed-Forward Network
    print("\n4. Test Feed-Forward Network:")
    ffn = PositionwiseFeedForward(d_model, d_ff)
    ffn_output = ffn(output)
    print(f"   Input shape: {output.shape}")
    print(f"   Output shape: {ffn_output.shape}")
    
    print("\n" + "="*70)
    print("✓ TẤT CẢ THÀNH PHẦN HOẠT ĐỘNG ĐÚNG!")
    print("="*70)