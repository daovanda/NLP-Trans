import re
import json
import pickle
import os
import pandas as pd
from tqdm import tqdm
from tokenizer_sentencepiece import (
    SentencePieceTokenizer,
    train_tokenizers_from_csv
)

# 1. LOAD DATASET
def load_csv_data(csv_path, vi_column='vi', en_column='en', 
                  train_ratio=0.8, val_ratio=0.1, chunk_size=50000):
    import pandas as pd, json, os
    from datasets import load_dataset

    print(f" Streaming CSV từ: {csv_path}")
    os.makedirs('tmp_splits', exist_ok=True)

    train_file = open('tmp_splits/train.jsonl', 'w', encoding='utf-8')
    val_file   = open('tmp_splits/val.jsonl', 'w', encoding='utf-8')
    test_file  = open('tmp_splits/test.jsonl', 'w', encoding='utf-8')

    total_rows = sum(1 for _ in open(csv_path, 'r', encoding='utf-8')) - 1
    train_cut = int(total_rows * train_ratio)
    val_cut   = int(total_rows * (train_ratio + val_ratio))
    idx = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk = chunk.dropna(subset=[vi_column, en_column])

        for _, row in chunk.iterrows():
            idx += 1
            obj = {"translation": {
                "vi": str(row[vi_column]).strip(),
                "en": str(row[en_column]).strip()
            }}
            line = json.dumps(obj, ensure_ascii=False) + "\n"

            if idx <= train_cut: train_file.write(line)
            elif idx <= val_cut: val_file.write(line)
            else: test_file.write(line)

        del chunk

    train_file.close(); val_file.close(); test_file.close()

    print(" Load dataset từ jsonl (không vào RAM)")
    return load_dataset("json", data_files={
        "train": "tmp_splits/train.jsonl",
        "validation": "tmp_splits/val.jsonl",
        "test": "tmp_splits/test.jsonl"
    })


def load_parallel_txt_files(train_vi_path, train_en_path, 
                            test_vi_path, test_en_path,
                            val_ratio=0.1, shuffle=True, seed=42):
    from datasets import load_dataset
    
    print("\n" + "="*70)
    print(" LOADING PARALLEL TEXT FILES")
    print("="*70)
    
    print("\n Reading train files...")
    with open(train_vi_path, 'r', encoding='utf-8') as f:
        train_vi = [line.strip() for line in f if line.strip()]
    with open(train_en_path, 'r', encoding='utf-8') as f:
        train_en = [line.strip() for line in f if line.strip()]
    
    print(f"   Train VI: {len(train_vi):,} lines")
    print(f"   Train EN: {len(train_en):,} lines")
    
    if len(train_vi) != len(train_en):
        raise ValueError(f"❌ Train files: Số dòng không khớp! VI: {len(train_vi)}, EN: {len(train_en)}")
    
    print("\nReading test files...")
    with open(test_vi_path, 'r', encoding='utf-8') as f:
        test_vi = [line.strip() for line in f if line.strip()]
    with open(test_en_path, 'r', encoding='utf-8') as f:
        test_en = [line.strip() for line in f if line.strip()]
    
    print(f"   Test VI: {len(test_vi):,} lines")
    print(f"   Test EN: {len(test_en):,} lines")
    
    if len(test_vi) != len(test_en):
        raise ValueError(f"❌ Test files: Số dòng không khớp! VI: {len(test_vi)}, EN: {len(test_en)}")
    
    print(f"\n Splitting validation ({val_ratio*100:.0f}% of train)...")
    
    total_train = len(train_vi)
    val_size = int(total_train * val_ratio)
    train_size = total_train - val_size
    
    if shuffle:
        print("    Shuffling train data để đảm bảo train/val có phân bố tương đồng...")
        import random
        random.seed(seed)
        
        indices = list(range(total_train))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        print(f"    Shuffled {total_train:,} pairs (seed={seed})")
    else:
        print("    No shuffle - sử dụng split tuần tự")
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_train))
    
    print(f"      → Train: {len(train_indices):,} pairs")
    print(f"      → Val: {len(val_indices):,} pairs")
    
    os.makedirs('tmp_splits', exist_ok=True)
    
    train_file = open('tmp_splits/train.jsonl', 'w', encoding='utf-8')
    for idx in train_indices:
        obj = {"translation": {"vi": train_vi[idx], "en": train_en[idx]}}
        train_file.write(json.dumps(obj, ensure_ascii=False) + "\n")
    train_file.close()
    
    val_file = open('tmp_splits/val.jsonl', 'w', encoding='utf-8')
    for idx in val_indices:
        obj = {"translation": {"vi": train_vi[idx], "en": train_en[idx]}}
        val_file.write(json.dumps(obj, ensure_ascii=False) + "\n")
    val_file.close()
    
    test_file = open('tmp_splits/test.jsonl', 'w', encoding='utf-8')
    for vi, en in zip(test_vi, test_en):
        obj = {"translation": {"vi": vi, "en": en}}
        test_file.write(json.dumps(obj, ensure_ascii=False) + "\n")
    test_file.close()
    
    print(f"   Test: {len(test_vi):,} pairs")
    
    print("\n Loading as HuggingFace dataset...")
    
    dataset = load_dataset("json", data_files={
        "train": "tmp_splits/train.jsonl",
        "validation": "tmp_splits/val.jsonl",
        "test": "tmp_splits/test.jsonl"
    })
    
    print(" Dataset loaded successfully!")
    
    return dataset
# 2. CLEANING
def clean_text(text: str, lang: str = 'vi') -> str:
    """Làm sạch văn bản"""
    if not text or not isinstance(text, str):
        return ""
    
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def is_valid_pair(vi_text, en_text, min_ratio=0.4, max_ratio=3.0):
    """Kiểm tra cặp câu có hợp lệ không"""
    lv, le = len(vi_text.split()), len(en_text.split())
    
    if lv < 2 or le < 2 or lv > 100 or le > 100:
        return False
    
    ratio = max(lv, le) / max(min(lv, le), 1)
    if ratio > max_ratio:
        return False
    
    if re.search(r'(.)\1{10,}', vi_text) or re.search(r'(.)\1{10,}', en_text):
        return False
    
    if len(set(vi_text.split())) < 2 or len(set(en_text.split())) < 2:
        return False
    
    return True

def clean_dataset(dataset):
    """Clean dataset và lưu thành jsonl"""
    import json, os
    os.makedirs('tmp_clean', exist_ok=True)

    paths = {
        'train': open('tmp_clean/train.jsonl', 'w', encoding='utf-8'),
        'validation': open('tmp_clean/val.jsonl', 'w', encoding='utf-8'),
        'test': open('tmp_clean/test.jsonl', 'w', encoding='utf-8'),
    }
    
    stats = {
        'train': {'total': 0, 'filtered': 0},
        'validation': {'total': 0, 'filtered': 0},
        'test': {'total': 0, 'filtered': 0}
    }

    for split in ['train', 'validation', 'test']:
        print(f"🔄 Cleaning {split} ...")
        for ex in tqdm(dataset[split], desc=f"Cleaning {split}"):
            stats[split]['total'] += 1
            
            vi = clean_text(ex['translation']['vi'])
            en = clean_text(ex['translation']['en'])
            
            if not vi or not en:
                stats[split]['filtered'] += 1
                continue
            
            if not is_valid_pair(vi, en):
                stats[split]['filtered'] += 1
                continue
            
            paths[split].write(json.dumps({"vi": vi, "en": en}, ensure_ascii=False) + "\n")

    for f in paths.values(): 
        f.close()
    
    print("\nCLEANING STATISTICS:")
    for split in ['train', 'validation', 'test']:
        total = stats[split]['total']
        filtered = stats[split]['filtered']
        kept = total - filtered
        print(f"  {split}:")
        print(f"    Total: {total:,}")
        print(f"    Filtered: {filtered:,} ({filtered/total*100:.2f}%)")
        print(f"    Kept: {kept:,}")

    from datasets import load_dataset
    ds = load_dataset("json", data_files={
        "train": "tmp_clean/train.jsonl",
        "validation": "tmp_clean/val.jsonl",
        "test": "tmp_clean/test.jsonl",
    })

    cleaned = {
        split: [{"vi": x["vi"], "en": x["en"]} for x in ds[split]]
        for split in ['train', 'validation', 'test']
    }
    return cleaned
    
# 3. PREPARE DATA WITH SENTENCEPIECE

def prepare_data_with_sentencepiece(cleaned_data, vi_tokenizer, en_tokenizer, max_len=100):
    """
    Chuẩn bị dữ liệu sử dụng SentencePiece tokenizers
    """
    print("\n" + "="*70)
    print("CHUẨN BỊ DỮ LIỆU VỚI SENTENCEPIECE")
    print("="*70)
    
    processed_data = {}
    
    stats = {
        'train': {'total': 0, 'skipped': 0},
        'validation': {'total': 0, 'skipped': 0},
        'test': {'total': 0, 'skipped': 0}
    }
    
    for split in ['train', 'validation', 'test']:
        print(f"\nĐang xử lý {split} set...")
        
        src_data = []
        tgt_data = []
        
        for item in tqdm(cleaned_data[split], desc=f"Processing {split}"):
            stats[split]['total'] += 1
            
            if not item.get('vi') or not item.get('en'):
                stats[split]['skipped'] += 1
                continue
            
            vi_text = item['vi'].strip()
            en_text = item['en'].strip()
            
            if not vi_text or not en_text:
                stats[split]['skipped'] += 1
                continue
            
            try:
                src_tokens = vi_tokenizer.encode(vi_text, add_bos=True, add_eos=True)
                tgt_tokens = en_tokenizer.encode(en_text, add_bos=True, add_eos=True)
                
                if len(src_tokens) <= 2 or len(tgt_tokens) <= 2:
                    stats[split]['skipped'] += 1
                    continue
                
                if len(src_tokens) > max_len:
                    src_tokens = src_tokens[:max_len-1] + [vi_tokenizer.EOS_IDX]
                if len(tgt_tokens) > max_len:
                    tgt_tokens = tgt_tokens[:max_len-1] + [en_tokenizer.EOS_IDX]
                
                src_data.append(src_tokens)
                tgt_data.append(tgt_tokens)
                
            except Exception as e:
                print(f"\n Warning: Failed to encode:")
                print(f"   VI: {vi_text[:50]}...")
                print(f"   EN: {en_text[:50]}...")
                print(f"   Error: {e}")
                stats[split]['skipped'] += 1
                continue
        
        processed_data[split] = {
            'src': src_data,
            'tgt': tgt_data
        }
        
        print(f"  Số mẫu giữ lại: {len(src_data):,}")
        print(f"  Số mẫu bỏ qua: {stats[split]['skipped']:,}")
        
        if src_data:
            avg_src_len = sum(len(s) for s in src_data) / len(src_data)
            avg_tgt_len = sum(len(t) for t in tgt_data) / len(tgt_data)
            print(f"  Avg source length: {avg_src_len:.1f}")
            print(f"  Avg target length: {avg_tgt_len:.1f}")
    
    print("\n" + "="*70)
    print(" TOKENIZATION SUMMARY")
    print("="*70)
    for split in ['train', 'validation', 'test']:
        total = stats[split]['total']
        skipped = stats[split]['skipped']
        kept = total - skipped
        print(f"\n{split}:")
        print(f"  Total: {total:,}")
        print(f"  Skipped: {skipped:,} ({skipped/total*100:.2f}%)")
        print(f"  Kept: {kept:,} ({kept/total*100:.2f}%)")
    
    return processed_data

# 4. SAVE/LOAD

def save_data_v2(cleaned_data, vi_tokenizer, en_tokenizer, processed_data, output_dir='../data/processed'):
    """
    Lưu dữ liệu và tokenizers
    """
    print("\n" + "="*70)
    print("LƯU DỮ LIỆU")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSaving cleaned_data.json...")
    with open(os.path.join(output_dir, 'cleaned_data.json'), 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    print(" Đã lưu cleaned_data.json")
    
    print("\n Saving processed_data.pkl...")
    processed_path = os.path.join(output_dir, 'processed_data.pkl')
    with open(processed_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    if os.path.exists(processed_path):
        file_size = os.path.getsize(processed_path) / (1024 * 1024)
        print(f" Đã lưu processed_data.pkl ({file_size:.2f} MB)")
        
        with open(processed_path, 'rb') as f:
            verify_data = pickle.load(f)
        print(f" Verified: {len(verify_data['train']['src']):,} train samples")
    else:
        print("ERROR: processed_data.pkl was NOT created!")
    
    print("\n Saving tokenizer_info.json...")
    tokenizer_info = {
        'vi_model_path': os.path.join(output_dir, 'vi_sp.model'),
        'en_model_path': os.path.join(output_dir, 'en_sp.model'),
        'vi_vocab_size': len(vi_tokenizer),
        'en_vocab_size': len(en_tokenizer),
        'pad_idx': 0,
        'unk_idx': 1,
        'bos_idx': 2,
        'eos_idx': 3
    }
    
    with open(os.path.join(output_dir, 'tokenizer_info.json'), 'w') as f:
        json.dump(tokenizer_info, f, indent=2)
    print(" Đã lưu tokenizer_info.json")
    
    print(f"\n Tất cả files đã được lưu vào: {output_dir}")
    print("\n Files created:")
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        if os.path.isfile(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {filename} ({size_mb:.2f} MB)")

def load_tokenizers(data_dir='../data/processed'):
    """
    Load SentencePiece tokenizers
    """
    vi_model_path = os.path.join(data_dir, 'vi_sp.model')
    en_model_path = os.path.join(data_dir, 'en_sp.model')
    
    print("Loading tokenizers...")
    vi_tokenizer = SentencePieceTokenizer(vi_model_path)
    en_tokenizer = SentencePieceTokenizer(en_model_path)
    
    return vi_tokenizer, en_tokenizer

# 5. TRAIN TOKENIZERS FROM TXT FILES

def train_tokenizers_from_txt(
    vi_txt_path,
    en_txt_path,
    output_dir,
    vi_vocab_size=32000,
    en_vocab_size=32000,
    model_type='bpe'
):
    import sentencepiece as spm
    
    print("\n Training SentencePiece tokenizers from .txt files...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n Training Vietnamese tokenizer (vocab_size={vi_vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=vi_txt_path,
        model_prefix=os.path.join(output_dir, 'vi_sp'),
        vocab_size=vi_vocab_size,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        character_coverage=0.9995,
        num_threads=os.cpu_count()
    )
    print("Vietnamese tokenizer trained!")
    
    print(f"\nTraining English tokenizer (vocab_size={en_vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=en_txt_path,
        model_prefix=os.path.join(output_dir, 'en_sp'),
        vocab_size=en_vocab_size,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        character_coverage=0.9995,
        num_threads=os.cpu_count()
    )
    print("English tokenizer trained!")
    
    vi_tokenizer = SentencePieceTokenizer(os.path.join(output_dir, 'vi_sp.model'))
    en_tokenizer = SentencePieceTokenizer(os.path.join(output_dir, 'en_sp.model'))
    
    print(f"\n Tokenizer info:")
    print(f"   VI vocab size: {len(vi_tokenizer):,}")
    print(f"   EN vocab size: {len(en_tokenizer):,}")
    
    return vi_tokenizer, en_tokenizer

# 6. MAIN PIPELINE V2

def main_v2(
    csv_path,
    csv_vi_col='vi',
    csv_en_col='en',
    output_dir='../data/processed',
    vi_vocab_size=32000,
    en_vocab_size=32000,
    retrain_tokenizer=True,
    sample_size=None
):
    """
    Pipeline xử lý dữ liệu hoàn chỉnh với SentencePiece
    """
    
    print("\n" + "="*70)
    print("DATA PREPROCESSING PIPELINE V2 - FIXED VERSION")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("STEP 1: TOKENIZERS")
    print("="*70)
    
    vi_model_path = os.path.join(output_dir, 'vi_sp.model')
    en_model_path = os.path.join(output_dir, 'en_sp.model')
    
    if retrain_tokenizer or not (os.path.exists(vi_model_path) and os.path.exists(en_model_path)):
        print("🔧 Training new SentencePiece tokenizers...")
        vi_tokenizer, en_tokenizer = train_tokenizers_from_csv(
            csv_path=csv_path,
            vi_column=csv_vi_col,
            en_column=csv_en_col,
            output_dir=output_dir,
            vi_vocab_size=vi_vocab_size,
            en_vocab_size=en_vocab_size,
            sample_size=sample_size,
            model_type='bpe'
        )
    else:
        print(" Loading existing tokenizers...")
        vi_tokenizer = SentencePieceTokenizer(vi_model_path)
        en_tokenizer = SentencePieceTokenizer(en_model_path)
    
    print("\n" + "="*70)
    print("STEP 2: LOAD & CLEAN DATA")
    print("="*70)
    
    dataset = load_csv_data(csv_path, csv_vi_col, csv_en_col)
    
    print("\n THỐNG KÊ DỮ LIỆU:")
    for split in dataset.keys():
        print(f"  - {split}: {len(dataset[split])} cặp câu")
    
    cleaned_data = clean_dataset(dataset)
    
    print("\n" + "="*70)
    print("STEP 3: TOKENIZE DATA WITH SENTENCEPIECE")
    print("="*70)
    
    processed_data = prepare_data_with_sentencepiece(
        cleaned_data, vi_tokenizer, en_tokenizer
    )
    
    print("\n" + "="*70)
    print("STEP 4: SAVE ALL DATA")
    print("="*70)
    
    save_data_v2(cleaned_data, vi_tokenizer, en_tokenizer, processed_data, output_dir)
    
    print("\n" + "="*70)
    print("STEP 5: VERIFICATION")
    print("="*70)
    
    pkl_path = os.path.join(output_dir, 'processed_data.pkl')
    if os.path.exists(pkl_path):
        print(f"\n processed_data.pkl exists!")
        print(f"   Size: {os.path.getsize(pkl_path) / (1024*1024):.2f} MB")
        
        with open(pkl_path, 'rb') as f:
            verify_data = pickle.load(f)
        
        print(f"\n Verified content:")
        for split in ['train', 'validation', 'test']:
            src_count = len(verify_data[split]['src'])
            tgt_count = len(verify_data[split]['tgt'])
            print(f"   {split}: {src_count:,} samples")
            
            all_src_tokens = [token for seq in verify_data[split]['src'] for token in seq]
            all_tgt_tokens = [token for seq in verify_data[split]['tgt'] for token in seq]
            
            max_src = max(all_src_tokens) if all_src_tokens else 0
            max_tgt = max(all_tgt_tokens) if all_tgt_tokens else 0
            
            print(f"      Max source token: {max_src} (vocab: {len(vi_tokenizer)})")
            print(f"      Max target token: {max_tgt} (vocab: {len(en_tokenizer)})")
            
            if max_src >= len(vi_tokenizer):
                print(f"       ERROR: Token {max_src} >= vocab size {len(vi_tokenizer)}")
            elif max_tgt >= len(en_tokenizer):
                print(f"      ERROR: Token {max_tgt} >= vocab size {len(en_tokenizer)}")
            else:
                print(f"      z All tokens within range")
    else:
        print(f"\n ERROR: processed_data.pkl NOT found at {pkl_path}")
    
    print("\n" + "="*70)
    print("HOÀN THÀNH XỬ LÝ DỮ LIỆU!")
    print("="*70)
    print(f"\n Tóm tắt:")
    print(f"  - Kích thước từ điển VI: {len(vi_tokenizer):,}")
    print(f"  - Kích thước từ điển EN: {len(en_tokenizer):,}")
    print(f"  - Train samples: {len(processed_data['train']['src']):,}")
    print(f"  - Validation samples: {len(processed_data['validation']['src']):,}")
    print(f"  - Test samples: {len(processed_data['test']['src']):,}")
    
    print(f"\n So sánh với vocab cũ:")
    print(f"  - Cũ: 174,608 (VI) + 251,853 (EN) = 426,461 tokens")
    print(f"  - Mới: {len(vi_tokenizer):,} (VI) + {len(en_tokenizer):,} (EN) = {len(vi_tokenizer) + len(en_tokenizer):,} tokens")
    print(f"  - Giảm: {(1 - (len(vi_tokenizer) + len(en_tokenizer)) / 426461) * 100:.1f}%")
    
    print(f"\n Ưu điểm SentencePiece:")
    print(f"   Vocab nhỏ hơn 5-10x → Model nhỏ hơn, train nhanh hơn")
    print(f"   Xử lý được từ mới (NO UNK tokens!)")
    print(f"   Tốt cho tiếng Việt có dấu")
    print(f"   Không cần retrain tokenizer khi có data mới")
    print(f"   Đã loại bỏ emoji, ký tự Nhật/Hàn/Trung")
    
    print(f"\n Files to upload to Kaggle:")
    print(f"  1. vi_sp.model")
    print(f"  2. en_sp.model")
    print(f"  3. cleaned_data.json (for reference)")
    print(f"  4. processed_data.pkl (NEW - with SentencePiece IDs)")
    print(f"  5. tokenizer_info.json")

# 7. MAIN PIPELINE FOR MEDICAL DATA (TXT FILES)

def main_txt_pipeline(
    train_vi_path,
    train_en_path,
    test_vi_path,
    test_en_path,
    output_dir='../data/processed_medical',
    vi_vocab_size=32000,
    en_vocab_size=32000,
    val_ratio=0.1,
    shuffle=True,
    seed=42,
    retrain_tokenizer=True
):
    
    print("\n" + "="*70)
    print(" MEDICAL DATA PREPROCESSING PIPELINE")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # STEP 1: Load data từ .txt files
    dataset = load_parallel_txt_files(
        train_vi_path, train_en_path,
        test_vi_path, test_en_path,
        val_ratio=val_ratio,
        shuffle=shuffle,
        seed=seed
    )
    
    # Stats
    print("\n THỐNG KÊ DỮ LIỆU:")
    for split in dataset.keys():
        print(f"  - {split}: {len(dataset[split])} cặp câu")
    
    # STEP 2: Train/Load Tokenizers
    print("\n" + "="*70)
    print("STEP 2: TOKENIZERS")
    print("="*70)
    
    vi_model_path = os.path.join(output_dir, 'vi_sp.model')
    en_model_path = os.path.join(output_dir, 'en_sp.model')
    
    if retrain_tokenizer or not (os.path.exists(vi_model_path) and os.path.exists(en_model_path)):
        print("🔧 Training new SentencePiece tokenizers từ raw text...")
        
        vi_tokenizer, en_tokenizer = train_tokenizers_from_txt(
            vi_txt_path=train_vi_path,
            en_txt_path=train_en_path,
            output_dir=output_dir,
            vi_vocab_size=vi_vocab_size,
            en_vocab_size=en_vocab_size,
            model_type='bpe'
        )
    else:
        print("Loading existing tokenizers...")
        vi_tokenizer = SentencePieceTokenizer(vi_model_path)
        en_tokenizer = SentencePieceTokenizer(en_model_path)
    
    # STEP 3: Clean Data
    print("\n" + "="*70)
    print("STEP 3: CLEAN DATA")
    print("="*70)
    
    cleaned_data = clean_dataset(dataset)
    
    # STEP 4: Tokenize
    print("\n" + "="*70)
    print("STEP 4: TOKENIZE")
    print("="*70)
    
    processed_data = prepare_data_with_sentencepiece(
        cleaned_data, vi_tokenizer, en_tokenizer
    )
    
    # STEP 5: Save
    print("\n" + "="*70)
    print("STEP 5: SAVE")
    print("="*70)
    
    save_data_v2(cleaned_data, vi_tokenizer, en_tokenizer, processed_data, output_dir)
    
    # STEP 6: Verification
    print("\n" + "="*70)
    print("STEP 6: VERIFICATION")
    print("="*70)
    
    pkl_path = os.path.join(output_dir, 'processed_data.pkl')
    if os.path.exists(pkl_path):
        print(f"\n processed_data.pkl exists!")
        print(f"   Size: {os.path.getsize(pkl_path) / (1024*1024):.2f} MB")
        
        with open(pkl_path, 'rb') as f:
            verify_data = pickle.load(f)
        
        print(f"\n Verified content:")
        for split in ['train', 'validation', 'test']:
            print(f"   {split}: {len(verify_data[split]['src']):,} samples")
    
    print("\n" + "="*70)
    print("HOÀN TẤT! ")
    print("="*70)
    
    return processed_data, vi_tokenizer, en_tokenizer


# ============================================================================
# 8. HELPER FUNCTION: QUICK LOAD
# ============================================================================

def quick_load(data_dir='../data/processed'):
    """
    Hàm tiện ích để load nhanh processed data và tokenizers
    """
    
    print(" Quick loading data and tokenizers...")
    
    # Load processed data
    with open(os.path.join(data_dir, 'processed_data.pkl'), 'rb') as f:
        processed_data = pickle.load(f)
    
    # Load tokenizers
    vi_tokenizer, en_tokenizer = load_tokenizers(data_dir)
    
    print("Loaded successfully!")
    print(f"   Train: {len(processed_data['train']['src']):,} samples")
    print(f"   Val: {len(processed_data['validation']['src']):,} samples")
    print(f"   Test: {len(processed_data['test']['src']):,} samples")
    print(f"   VI vocab: {len(vi_tokenizer):,}")
    print(f"   EN vocab: {len(en_tokenizer):,}")
    
    return processed_data, vi_tokenizer, en_tokenizer

# 9. MAIN

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Data Preprocessing V2 with SentencePiece (CSV or TXT)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['csv', 'txt'], required=True,
                       help='Mode: csv (for CSV files) or txt (for parallel .txt files)')
    
    # CSV mode arguments
    parser.add_argument('--csv_path', type=str,
                       help='[CSV mode] Đường dẫn file CSV')
    parser.add_argument('--vi_col', type=str, default='vi',
                       help='[CSV mode] Tên cột tiếng Việt')
    parser.add_argument('--en_col', type=str, default='en',
                       help='[CSV mode] Tên cột tiếng Anh')
    
    # TXT mode arguments
    parser.add_argument('--train_vi', type=str,
                       help='[TXT mode] Đường dẫn train.vi.txt')
    parser.add_argument('--train_en', type=str,
                       help='[TXT mode] Đường dẫn train.en.txt')
    parser.add_argument('--test_vi', type=str,
                       help='[TXT mode] Đường dẫn test.vi.txt')
    parser.add_argument('--test_en', type=str,
                       help='[TXT mode] Đường dẫn test.en.txt')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='[TXT mode] Tỷ lệ validation (0.1 = 10%%)')
    parser.add_argument('--no_shuffle', action='store_true',
                       help='[TXT mode] Không shuffle khi split validation')
    parser.add_argument('--seed', type=int, default=42,
                       help='[TXT mode] Random seed cho shuffle')
    
    # Common arguments
    parser.add_argument('--output_dir', type=str, default='../data/processed',
                       help='Thư mục output')
    parser.add_argument('--vi_vocab_size', type=int, default=32000,
                       help='Vocab size tiếng Việt')
    parser.add_argument('--en_vocab_size', type=int, default=32000,
                       help='Vocab size tiếng Anh')
    parser.add_argument('--no_retrain', action='store_true',
                       help='Không train lại tokenizer (dùng có sẵn)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='[CSV mode] Lấy N dòng đầu để test')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'csv':
        if not args.csv_path:
            parser.error("--csv_path is required for CSV mode")
        
        main_v2(
            csv_path=args.csv_path,
            csv_vi_col=args.vi_col,
            csv_en_col=args.en_col,
            output_dir=args.output_dir,
            vi_vocab_size=args.vi_vocab_size,
            en_vocab_size=args.en_vocab_size,
            retrain_tokenizer=not args.no_retrain,
            sample_size=args.sample_size
        )
    
    elif args.mode == 'txt':
        if not all([args.train_vi, args.train_en, args.test_vi, args.test_en]):
            parser.error("--train_vi, --train_en, --test_vi, --test_en are required for TXT mode")
        
        main_txt_pipeline(
            train_vi_path=args.train_vi,
            train_en_path=args.train_en,
            test_vi_path=args.test_vi,
            test_en_path=args.test_en,
            output_dir=args.output_dir,
            vi_vocab_size=args.vi_vocab_size,
            en_vocab_size=args.en_vocab_size,
            val_ratio=args.val_ratio,
            shuffle=not args.no_shuffle,
            seed=args.seed,
            retrain_tokenizer=not args.no_retrain
        )