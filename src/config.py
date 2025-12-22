"""
CONFIGURATION FILE - OPTIMIZED FOR GOOGLE COLAB
Cấu hình tối ưu cho training trong thời gian giới hạn
"""

import torch
from pathlib import Path

class Config:
    """Global configuration tối ưu cho Colab"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    LOGS_DIR = PROJECT_ROOT / 'logs'
    
    # Data
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    VI_VOCAB_PATH = PROCESSED_DATA_DIR / 'vi_vocab.pkl'
    EN_VOCAB_PATH = PROCESSED_DATA_DIR / 'en_vocab.pkl'
    PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / 'processed_data.pkl'
    
    # Model - USE TINY/SMALL FOR FASTER TRAINING
    MODEL_SIZE = 'tiny'  # 'tiny' nhanh nhất, 'small' cân bằng, 'base' chậm
    PAD_IDX = 0
    
    # Training - OPTIMIZED FOR COLAB 2-HOUR LIMIT
    NUM_EPOCHS = 20
    BATCH_SIZE = 64  # Tăng batch size với mixed precision
    WARMUP_STEPS = 2000  # Giảm warmup steps
    LABEL_SMOOTHING = 0.1
    
    # CRITICAL: Lưu checkpoint theo batch
    SAVE_EVERY_BATCHES = 500  # Lưu mỗi 500 batches (~5-10 phút)
    SAVE_EVERY_EPOCHS = 1     # Lưu cuối mỗi epoch
    
    # Mixed Precision - IMPORTANT: Tăng tốc 2x
    USE_AMP = True  # Automatic Mixed Precision
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # DataLoader - Colab hỗ trợ multiprocessing
    NUM_WORKERS = 2
    
    # Evaluation
    BEAM_SIZE = 5
    MAX_DECODE_LEN = 100

    @classmethod
    def print_config(cls):
        """In ra cấu hình hiện tại"""
        print("="*70)
        print("⚙️  TRAINING CONFIGURATION (OPTIMIZED FOR COLAB)")
        print("="*70)
        print(f"Model Size:           {cls.MODEL_SIZE}")
        print(f"Batch Size:           {cls.BATCH_SIZE}")
        print(f"Epochs:               {cls.NUM_EPOCHS}")
        print(f"Mixed Precision:      {cls.USE_AMP}")
        print(f"Save Every Batches:   {cls.SAVE_EVERY_BATCHES}")
        print(f"Device:               {cls.DEVICE}")
        print("="*70)
        if cls.USE_AMP and torch.cuda.is_available():
            print("✅ Mixed Precision enabled - Training will be ~2x faster!")
        if cls.SAVE_EVERY_BATCHES <= 1000:
            print(f"✅ Checkpoint every {cls.SAVE_EVERY_BATCHES} batches - Safe for 2h limit!")
        print("="*70)