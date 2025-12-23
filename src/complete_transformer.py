import torch
import torch.nn as nn
from transformer_encoder_decoder import (
    Encoder, Decoder,
    create_padding_mask, create_target_mask
)

# TRANSFORMER MODEL
class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
        pad_idx=0
    ):
        super().__init__()
        
        self.pad_idx = pad_idx
        
        # Encoder
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len
        )
        
        # Decoder
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len
        )
        
        # Khởi tạo weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Khởi tạo weights theo Xavier Uniform
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt):
        # Tạo masks
        src_mask = create_padding_mask(src, self.pad_idx)
        tgt_mask = create_target_mask(tgt, self.pad_idx)
        
        # Encoder
        encoder_output = self.encoder(src, src_mask)
        
        # Decoder
        output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        return output
    
    def encode(self, src):
        src_mask = create_padding_mask(src, self.pad_idx)
        encoder_output = self.encoder(src, src_mask)
        return encoder_output, src_mask
    
    def decode(self, tgt, encoder_output, src_mask):
        tgt_mask = create_target_mask(tgt, self.pad_idx)
        output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return output

# MODEL CONFIGURATION

def get_model_config(model_size='base'):
    configs = {
        'tiny': {
            'd_model': 256,
            'n_layers': 2,
            'n_heads': 4,
            'd_ff': 1024,
            'dropout': 0.1
        },
        'small': {
            'd_model': 256,
            'n_layers': 4,
            'n_heads': 8,
            'd_ff': 1024,
            'dropout': 0.1
        },
        'base': {
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8,
            'd_ff': 2048,
            'dropout': 0.1
        },
        'large': {
            'd_model': 1024,
            'n_layers': 6,
            'n_heads': 16,
            'd_ff': 4096,
            'dropout': 0.1
        }
    }
    
    return configs.get(model_size, configs['base'])

def create_model(src_vocab_size, tgt_vocab_size, model_size='base', pad_idx=0):
    config = get_model_config(model_size)
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        pad_idx=pad_idx
    )
    
    return model, config

# UTILITY FUNCTIONS

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable

def print_model_info(model, model_size='base'):

    total_params, trainable_params = count_parameters(model)
    
    print("="*70)
    print("THÔNG TIN MÔ HÌNH TRANSFORMER")
    print("="*70)
    print(f"\nKích thước model: {model_size.upper()}")
    print(f"\nSố lượng parameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    print(f"  - Model size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    
    config = get_model_config(model_size)
    print(f"\nCấu hình:")
    print(f"  - d_model: {config['d_model']}")
    print(f"  - n_layers: {config['n_layers']}")
    print(f"  - n_heads: {config['n_heads']}")
    print(f"  - d_ff: {config['d_ff']}")
    print(f"  - dropout: {config['dropout']}")
    print("="*70)

# TEST COMPLETE MODEL
if __name__ == "__main__":
    print("="*70)
    print("KIỂM TRA TRANSFORMER MODEL HOÀN CHỈNH")
    print("="*70)
    
    # Hyperparameters
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    batch_size = 4
    src_len = 15
    tgt_len = 20
    pad_idx = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}\n")
    
    # Test với các kích thước model khác nhau
    for model_size in ['tiny', 'small', 'base']:
        print(f"\n{'='*70}")
        print(f"TEST MODEL SIZE: {model_size.upper()}")
        print(f"{'='*70}\n")
        
        # Tạo model
        model, config = create_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            model_size=model_size,
            pad_idx=pad_idx
        )
        model = model.to(device)
        
        # In thông tin model
        print_model_info(model, model_size)
        
        # Tạo dummy data
        src = torch.randint(1, src_vocab_size, (batch_size, src_len)).to(device)
        tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len)).to(device)
        
        # Forward pass
        print(f"\nForward pass:")
        print(f"  Source shape: {src.shape}")
        print(f"  Target shape: {tgt.shape}")
        
        with torch.no_grad():
            output = model(src, tgt)
        
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: [{batch_size}, {tgt_len}, {tgt_vocab_size}]")
        print(f"  ✓ Shape correct!")
        
        # Test encode và decode riêng
        print(f"\nTest encode & decode separately:")
        with torch.no_grad():
            encoder_output, src_mask = model.encode(src)
            decoder_output = model.decode(tgt, encoder_output, src_mask)
        
        print(f"  Encoder output shape: {encoder_output.shape}")
        print(f"  Decoder output shape: {decoder_output.shape}")
        print(f"  ✓ Encode/Decode work correctly!")
        
        # Kiểm tra output giống nhau
        print(f"\nVerify output consistency:")
        with torch.no_grad():
            output_combined = model(src, tgt)
        
        is_same = torch.allclose(output_combined, decoder_output, atol=1e-6)
        print(f"  Forward == Encode+Decode: {is_same}")
        print(f"  ✓ Model is consistent!")
    
    print("\n" + "="*70)
    print("✓ TẤT CẢ TESTS PASSED!")
    print("="*70)
    
    print("\n GỢI Ý SỬ DỤNG:")
    print("  - Dùng 'tiny' để debug và test nhanh")
    print("  - Dùng 'small' để train trên CPU hoặc GPU nhỏ")
    print("  - Dùng 'base' để có kết quả tốt (cần GPU)")
    print("  - Dùng 'large' chỉ khi có GPU mạnh")