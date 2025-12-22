"""
DATALOADER MODULE - TỐI ỨU HÓA BATCH PROCESSING (FIXED FOR WINDOWS)
Xử lý batch với dynamic padding để tăng tốc huấn luyện
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import numpy as np

# ============================================================================
# 1. CUSTOM DATASET
# ============================================================================

class TranslationDataset(Dataset):
    """
    Dataset class cho dữ liệu dịch máy
    """
    def __init__(self, src_data, tgt_data):
        """
        Args:
            src_data: List of source sequences (đã encode thành indices)
            tgt_data: List of target sequences (đã encode thành indices)
        """
        assert len(src_data) == len(tgt_data), "Source và Target phải có cùng số lượng mẫu"
        
        self.src_data = src_data
        self.tgt_data = tgt_data
        
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        """
        Trả về một cặp (source, target) dưới dạng tensor
        """
        src = torch.LongTensor(self.src_data[idx])
        tgt = torch.LongTensor(self.tgt_data[idx])
        
        return src, tgt

# ============================================================================
# 2. COLLATE FUNCTION - DYNAMIC PADDING
# ============================================================================

def collate_fn(batch, pad_idx=0):
    """
    Collate function với dynamic padding
    Chỉ pad đến độ dài câu dài nhất trong batch, không phải max_len cố định
    
    Args:
        batch: List of (src, tgt) tuples
        pad_idx: Index của padding token
        
    Returns:
        src_batch: Padded source sequences [batch_size, max_src_len]
        tgt_batch: Padded target sequences [batch_size, max_tgt_len]
        src_lengths: Độ dài thực của mỗi source sequence
        tgt_lengths: Độ dài thực của mỗi target sequence
    """
    # Tách source và target
    src_batch, tgt_batch = zip(*batch)
    
    # Lấy độ dài thực của mỗi sequence (trước khi pad)
    src_lengths = torch.LongTensor([len(s) for s in src_batch])
    tgt_lengths = torch.LongTensor([len(t) for t in tgt_batch])
    
    # Padding - pad_sequence tự động pad đến độ dài max trong batch
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    
    return src_batch, tgt_batch, src_lengths, tgt_lengths

# ============================================================================
# COLLATE WRAPPER - FIX CHO WINDOWS MULTIPROCESSING
# ============================================================================

class CollateWrapper:
    """
    Wrapper cho collate_fn để tránh lỗi pickle với lambda trên Windows
    """
    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        return collate_fn(batch, self.pad_idx)

# ============================================================================
# 3. TẠO DATALOADER
# ============================================================================

def create_dataloaders(processed_data, batch_size=32, num_workers=0):
    """
    Tạo DataLoader cho train, validation và test
    
    Args:
        processed_data: Dict chứa 'train', 'validation', 'test'
        batch_size: Kích thước batch
        num_workers: Số worker threads (set = 0 cho Windows)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n" + "="*70)
    print("TẠO DATALOADERS")
    print("="*70)
    
    # Tạo datasets
    train_dataset = TranslationDataset(
        processed_data['train']['src'],
        processed_data['train']['tgt']
    )
    
    val_dataset = TranslationDataset(
        processed_data['validation']['src'],
        processed_data['validation']['tgt']
    )
    
    test_dataset = TranslationDataset(
        processed_data['test']['src'],
        processed_data['test']['tgt']
    )
    
    print(f"\nKích thước datasets:")
    print(f"  - Train: {len(train_dataset)}")
    print(f"  - Validation: {len(val_dataset)}")
    print(f"  - Test: {len(test_dataset)}")
    
    # Tạo collate wrapper
    collate_wrapper = CollateWrapper(pad_idx=0)
    
    # Tạo dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_wrapper,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Chỉ pin_memory nếu có GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_wrapper,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_wrapper,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"\nSố batch mỗi epoch:")
    print(f"  - Train: {len(train_loader)}")
    print(f"  - Validation: {len(val_loader)}")
    print(f"  - Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

# ============================================================================
# 4. BUCKET SAMPLER - TỐI ƯU HƠN (OPTIONAL)
# ============================================================================

class BucketSampler(torch.utils.data.Sampler):
    """
    Sampler nhóm các câu có độ dài tương tự vào cùng batch
    Giảm thiểu padding, tăng tốc huấn luyện
    """
    def __init__(self, data_source, batch_size, sort_key=lambda x: len(x)):
        self.data_source = data_source
        self.batch_size = batch_size
        self.sort_key = sort_key
        
    def __iter__(self):
        # Lấy indices và độ dài
        indices = list(range(len(self.data_source)))
        lengths = [self.sort_key(self.data_source[i]) for i in indices]
        
        # Sort theo độ dài
        sorted_indices = [i for i, _ in sorted(zip(indices, lengths), key=lambda x: x[1])]
        
        # Chia thành các batch
        batches = [sorted_indices[i:i+self.batch_size] 
                  for i in range(0, len(sorted_indices), self.batch_size)]
        
        # Shuffle thứ tự các batch (không shuffle trong batch)
        np.random.shuffle(batches)
        
        # Flatten
        for batch in batches:
            for idx in batch:
                yield idx
    
    def __len__(self):
        return len(self.data_source)

def create_dataloaders_with_bucketing(processed_data, batch_size=32, num_workers=0):
    """
    Tạo DataLoader với BucketSampler để tối ưu padding
    
    IMPORTANT: num_workers phải = 0 trên Windows
    """
    print("\n" + "="*70)
    print("TẠO DATALOADERS VỚI BUCKET SAMPLING")
    print("="*70)
    
    # Tạo datasets
    train_dataset = TranslationDataset(
        processed_data['train']['src'],
        processed_data['train']['tgt']
    )
    
    val_dataset = TranslationDataset(
        processed_data['validation']['src'],
        processed_data['validation']['tgt']
    )
    
    test_dataset = TranslationDataset(
        processed_data['test']['src'],
        processed_data['test']['tgt']
    )
    
    # Tạo BucketSampler cho train
    train_sampler = BucketSampler(
        train_dataset.src_data,
        batch_size=batch_size,
        sort_key=lambda x: len(x)
    )
    
    # Tạo collate wrapper
    collate_wrapper = CollateWrapper(pad_idx=0)
    
    # Tạo dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_wrapper,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_wrapper,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_wrapper,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"✓ Đã tạo DataLoaders với Bucket Sampling")
    print(f"  Bucket Sampling giúp giảm padding, tăng tốc ~15-20%")
    if num_workers == 0:
        print(f"  ⚠️  num_workers=0 (Windows compatibility mode)")
    
    return train_loader, val_loader, test_loader

# ============================================================================
# 5. HELPER FUNCTIONS
# ============================================================================

def load_data_and_vocab():
    """
    Load dữ liệu và vocabulary đã xử lý
    """
    print("Đang load dữ liệu và vocabulary...")
    
    with open('../data/processed/processed_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    with open('../data/processed/vi_vocab.pkl', 'rb') as f:
        vi_vocab = pickle.load(f)
    
    with open('../data/processed/en_vocab.pkl', 'rb') as f:
        en_vocab = pickle.load(f)
    
    print("✓ Đã load xong!")
    
    return processed_data, vi_vocab, en_vocab

def test_dataloader(loader, vi_vocab, en_vocab, num_batches=2):
    """
    Test dataloader và xem dữ liệu
    """
    print("\n" + "="*70)
    print("KIỂM TRA DATALOADER")
    print("="*70)
    
    for i, (src, tgt, src_len, tgt_len) in enumerate(loader):
        if i >= num_batches:
            break
            
        print(f"\nBatch {i+1}:")
        print(f"  Source shape: {src.shape}")
        print(f"  Target shape: {tgt.shape}")
        print(f"  Source lengths: {src_len[:5].tolist()}...")
        print(f"  Target lengths: {tgt_len[:5].tolist()}...")
        
        # Decode ví dụ đầu tiên
        print(f"\n  Ví dụ đầu tiên trong batch:")
        src_text = vi_vocab.decode(src[0].tolist())
        tgt_text = en_vocab.decode(tgt[0].tolist())
        print(f"    VI: {src_text}")
        print(f"    EN: {tgt_text}")

# ============================================================================
# 6. MAIN
# ============================================================================

if __name__ == "__main__":
    # Load dữ liệu
    processed_data, vi_vocab, en_vocab = load_data_and_vocab()
    
    # Tạo dataloaders thông thường
    print("\n" + "="*70)
    print("PHƯƠNG ÁN 1: DATALOADER THÔNG THƯỜNG")
    print("="*70)
    train_loader, val_loader, test_loader = create_dataloaders(
        processed_data, 
        batch_size=32,
        num_workers=0  # 0 cho Windows
    )
    
    # Test
    test_dataloader(train_loader, vi_vocab, en_vocab, num_batches=2)
    
    # Tạo dataloaders với bucketing (khuyên dùng)
    print("\n" + "="*70)
    print("PHƯƠNG ÁN 2: DATALOADER VỚI BUCKET SAMPLING (KHUYÊN DÙNG)")
    print("="*70)
    train_loader_bucket, val_loader_bucket, test_loader_bucket = create_dataloaders_with_bucketing(
        processed_data,
        batch_size=32,
        num_workers=0  # 0 cho Windows
    )
    
    # Test
    test_dataloader(train_loader_bucket, vi_vocab, en_vocab, num_batches=2)
    
    print("\n" + "="*70)
    print("✓ HOÀN TẤT TẠO DATALOADER!")
    print("="*70)
    print("\nGợi ý:")
    print("  - Sử dụng batch_size=32 hoặc 64 tùy GPU")
    print("  - Sử dụng BucketSampler để tối ưu tốc độ")
    print("  - num_workers=0 trên Windows (multiprocessing issue)")
    print("  - num_workers=2-4 trên Linux/Mac để tăng tốc")