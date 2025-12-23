
import sentencepiece as spm
import os
import json
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm

# 0. TEXT CLEANING FOR TOKENIZER TRAINING

def is_valid_english_char(char):
    """Kiểm tra ký tự có phải English hợp lệ không"""
    if char.isalnum() and char.isascii():
        return True
    if char in ' .,!?;:\'"()-[]{}/@#$%^&*+=_~`|\\<>\n\t':
        return True
    return False

def is_valid_vietnamese_char(char):
    """Kiểm tra ký tự có phải Vietnamese hợp lệ không"""
    if char.isalnum() and char.isascii():
        return True
    vietnamese_chars = 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
    vietnamese_chars += vietnamese_chars.upper()
    if char in vietnamese_chars:
        return True
    if char in ' .,!?;:\'"()-[]{}/@#$%^&*+=_~`|\\<>\n\t':
        return True
    return False

def clean_text_for_tokenizer(text, lang='en'):
    """Làm sạch text trước khi train tokenizer - loại bỏ ký tự lạ"""
    if not text or not isinstance(text, str):
        return ""
    
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+')
    text = cjk_pattern.sub('', text)
    
    if lang == 'en':
        cleaned = ''.join(char if is_valid_english_char(char) else ' ' for char in text)
    else:
        cleaned = ''.join(char if is_valid_vietnamese_char(char) else ' ' for char in text)
    
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

def is_valid_sentence(text, lang='en', min_words=2, max_words=100):
    """Kiểm tra câu có hợp lệ không"""
    if not text or len(text.strip()) < 3:
        return False
    
    words = text.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    
    if re.search(r'(.)\1{10,}', text):
        return False
    
    if len(set(words)) < 2:
        return False
    
    return True

# 1. TRAIN SENTENCEPIECE MODEL

def train_sentencepiece_model(
    input_file,
    model_prefix,
    vocab_size=32000,
    model_type='bpe',
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
):
    """
    Train SentencePiece model từ file text
    """
    print(f"\n🔧 Training SentencePiece model: {model_prefix}")
    print(f"   Input: {input_file}")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Model type: {model_type}")
    
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
        user_defined_symbols=[],
        num_threads=os.cpu_count(),
        split_digits=True,
        byte_fallback=True,
    )
    
    model_path = f"{model_prefix}.model"
    vocab_path = f"{model_prefix}.vocab"
    
    print(f"   Model saved: {model_path}")
    print(f"   Vocab saved: {vocab_path}")
    
    return model_path, vocab_path


def prepare_corpus_file(data, output_file, language='vi'):
    """
    Chuẩn bị file corpus từ dữ liệu để train SentencePiece
    """
    print(f"\n Preparing corpus file: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if isinstance(data, dict):
            sentences = data['train']
        else:
            sentences = data
        
        for sentence in tqdm(sentences, desc="Writing corpus"):
            if isinstance(sentence, dict):
                text = sentence.get(language, '')
            else:
                text = sentence
            
            if text and text.strip():
                f.write(text.strip() + '\n')
    
    print(f"    Wrote {len(sentences)} sentences")
    return output_file


# 2. SENTENCEPIECE TOKENIZER WRAPPER

class SentencePieceTokenizer:

    def __init__(self, model_path):
        """
        Args:
            model_path: Đường dẫn đến .model file
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        self.PAD_IDX = self.sp.pad_id()
        self.UNK_IDX = self.sp.unk_id()
        self.SOS_IDX = self.sp.bos_id()
        self.EOS_IDX = self.sp.eos_id()
        
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.SOS_TOKEN = '<s>'
        self.EOS_TOKEN = '</s>'
        
        print(f" Loaded SentencePiece model: {model_path}")
        print(f"   Vocab size: {len(self.sp)}")
        print(f"   PAD_IDX: {self.PAD_IDX}")
        print(f"   UNK_IDX: {self.UNK_IDX}")
        print(f"   SOS_IDX: {self.SOS_IDX}")
        print(f"   EOS_IDX: {self.EOS_IDX}")
    
    def encode(self, text, add_bos=True, add_eos=True):
        """
        Encode text thành list of token IDs
        """
        if not text or not isinstance(text, str):
            if add_bos and add_eos:
                return [self.SOS_IDX, self.EOS_IDX]
            elif add_bos:
                return [self.SOS_IDX]
            elif add_eos:
                return [self.EOS_IDX]
            else:
                return []
        
        text = text.strip()
        
        if not text:
            if add_bos and add_eos:
                return [self.SOS_IDX, self.EOS_IDX]
            elif add_bos:
                return [self.SOS_IDX]
            elif add_eos:
                return [self.EOS_IDX]
            else:
                return []
        
        ids = self.sp.encode(text, out_type=int)
        
        if not ids:
            if add_bos and add_eos:
                return [self.SOS_IDX, self.EOS_IDX]
            elif add_bos:
                return [self.SOS_IDX]
            elif add_eos:
                return [self.EOS_IDX]
            else:
                return []
        
        if add_bos and ids[0] != self.SOS_IDX:
            ids = [self.SOS_IDX] + ids
        
        if add_eos and ids[-1] != self.EOS_IDX:
            ids = ids + [self.EOS_IDX]
        
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        """
        Decode list of IDs thành text
        """
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        
        if not ids:
            return ""
        
        if skip_special_tokens:
            ids = [
                idx for idx in ids 
                if idx not in [self.PAD_IDX, self.SOS_IDX, self.EOS_IDX]
            ]
        
        if not ids:
            return ""
        
        return self.sp.decode(ids)
    
    def encode_as_pieces(self, text):
        """
        Encode text thành list of subword pieces (để debug)
        """
        if not text or not isinstance(text, str):
            return []
        
        text = text.strip()
        if not text:
            return []
        
        return self.sp.encode(text, out_type=str)
    
    def __len__(self):
        """Trả về vocab size"""
        return len(self.sp)
    
    def get_vocab_size(self):
        """Trả về vocab size"""
        return len(self.sp)
    
    def save_config(self, config_path):
        """Lưu config của tokenizer"""
        config = {
            'vocab_size': len(self.sp),
            'pad_idx': self.PAD_IDX,
            'unk_idx': self.UNK_IDX,
            'sos_idx': self.SOS_IDX,
            'eos_idx': self.EOS_IDX,
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f" Saved tokenizer config: {config_path}")

# 3. PIPELINE: TRAIN TOKENIZERS Từ CSV

def train_tokenizers_from_csv(
    csv_path,
    vi_column='vi',
    en_column='en',
    output_dir='../data/processed',
    vi_vocab_size=32000,
    en_vocab_size=32000,
    sample_size=None,
    model_type='bpe'
):
    """
    Pipeline hoàn chỉnh: CSV → SentencePiece tokenizers
    """
    import pandas as pd
    
    print("="*70)
    print("🚀 TRAINING SENTENCEPIECE TOKENIZERS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    import csv

    print(f"\n Reading CSV with cleaning: {csv_path}")

    vi_corpus_file = os.path.join(output_dir, 'vi_corpus.txt')
    en_corpus_file = os.path.join(output_dir, 'en_corpus.txt')

    vi_count = 0
    en_count = 0
    
    with open(csv_path, "r", encoding="utf-8") as fin, \
        open(vi_corpus_file, "w", encoding="utf-8") as f_vi, \
        open(en_corpus_file, "w", encoding="utf-8") as f_en:

        reader = csv.DictReader(fin)
        row_count = 0

        for row in tqdm(reader, desc="Processing & cleaning CSV"):
            vi_text = str(row[vi_column]).strip()
            en_text = str(row[en_column]).strip()
            
            vi_cleaned = clean_text_for_tokenizer(vi_text, lang='vi')
            en_cleaned = clean_text_for_tokenizer(en_text, lang='en')
            
            if is_valid_sentence(vi_cleaned, lang='vi'):
                f_vi.write(vi_cleaned + "\n")
                vi_count += 1
            
            if is_valid_sentence(en_cleaned, lang='en'):
                f_en.write(en_cleaned + "\n")
                en_count += 1

            row_count += 1
            if sample_size and row_count >= sample_size:
                break

    print(f"    Total rows processed: {row_count}")
    print(f"    Vietnamese valid sentences: {vi_count}")
    print(f"    English valid sentences: {en_count}")
    print(f"    Vietnamese corpus: {vi_corpus_file}")
    print(f"    English corpus: {en_corpus_file}")

    print("\n" + "="*70)
    print("🇻🇳 TRAINING VIETNAMESE TOKENIZER")
    print("="*70)
    
    vi_model_prefix = os.path.join(output_dir, 'vi_sp')
    train_sentencepiece_model(
        input_file=vi_corpus_file,
        model_prefix=vi_model_prefix,
        vocab_size=vi_vocab_size,
        model_type=model_type,
        character_coverage=0.9995
    )
    
    print("\n" + "="*70)
    print("🇬🇧 TRAINING ENGLISH TOKENIZER")
    print("="*70)
    
    en_model_prefix = os.path.join(output_dir, 'en_sp')
    train_sentencepiece_model(
        input_file=en_corpus_file,
        model_prefix=en_model_prefix,
        vocab_size=en_vocab_size,
        model_type=model_type,
        character_coverage=0.9995
    )
    
    print("\n" + "="*70)
    print(" LOADING TOKENIZERS")
    print("="*70)
    
    vi_tokenizer = SentencePieceTokenizer(f"{vi_model_prefix}.model")
    en_tokenizer = SentencePieceTokenizer(f"{en_model_prefix}.model")
    
    vi_tokenizer.save_config(os.path.join(output_dir, 'vi_tokenizer_config.json'))
    en_tokenizer.save_config(os.path.join(output_dir, 'en_tokenizer_config.json'))
    
    print("\n" + "="*70)
    print(" TESTING TOKENIZERS")
    print("="*70)
    
    vi_test = "Xin chào, tôi là một sinh viên đang học về trí tuệ nhân tạo."
    vi_ids = vi_tokenizer.encode(vi_test)
    vi_pieces = vi_tokenizer.encode_as_pieces(vi_test)
    vi_decoded = vi_tokenizer.decode(vi_ids)
    
    print(f"\n🇻🇳 Vietnamese:")
    print(f"   Original: {vi_test}")
    print(f"   Pieces: {vi_pieces}")
    print(f"   IDs: {vi_ids[:20]}...")
    print(f"   Decoded: {vi_decoded}")
    
    en_test = "Hello, I am a student learning about artificial intelligence."
    en_ids = en_tokenizer.encode(en_test)
    en_pieces = en_tokenizer.encode_as_pieces(en_test)
    en_decoded = en_tokenizer.decode(en_ids)
    
    print(f"\n🇬🇧 English:")
    print(f"   Original: {en_test}")
    print(f"   Pieces: {en_pieces}")
    print(f"   IDs: {en_ids[:20]}...")
    print(f"   Decoded: {en_decoded}")
    
    print(f"\n Testing UNK handling:")
    unk_test = "Thisjsdfklsjdfkljsdfisacompletelynewword"
    unk_ids = en_tokenizer.encode(unk_test)
    unk_pieces = en_tokenizer.encode_as_pieces(unk_test)
    unk_decoded = en_tokenizer.decode(unk_ids)
    print(f"   Input: {unk_test}")
    print(f"   Pieces: {unk_pieces}")
    print(f"    No UNK! Broken into subwords")
    print(f"   Decoded: {unk_decoded}")
    
    print("\n" + "="*70)
    print(" HOÀN THÀNH TRAINING TOKENIZERS!")
    print("="*70)
    print(f"\n Summary:")
    print(f"   Vietnamese vocab: {len(vi_tokenizer):,}")
    print(f"   English vocab: {len(en_tokenizer):,}")
    print(f"   Model files: {output_dir}/vi_sp.model, {output_dir}/en_sp.model")
    print(f"\n Ưu điểm:")
    print(f"    Vocab nhỏ hơn 5-10x so với word-based")
    print(f"    Xử lý được từ mới (no UNK!)")
    print(f"    Tốt cho cả tiếng Việt có dấu")
    print(f"    Đã loại bỏ emoji, ký tự Nhật/Hàn/Trung")
    
    return vi_tokenizer, en_tokenizer

# 4. MAIN - VÍ DỤ SỬ DỤNG

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Đường dẫn file CSV')
    parser.add_argument('--vi_col', type=str, default='vi',
                       help='Tên cột tiếng Việt')
    parser.add_argument('--en_col', type=str, default='en',
                       help='Tên cột tiếng Anh')
    parser.add_argument('--output_dir', type=str, default='../data/processed',
                       help='Thư mục output')
    parser.add_argument('--vi_vocab_size', type=int, default=32000,
                       help='Vocab size tiếng Việt')
    parser.add_argument('--en_vocab_size', type=int, default=32000,
                       help='Vocab size tiếng Anh')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Lấy N dòng đầu để test (None = hết)')
    parser.add_argument('--model_type', type=str, default='bpe',
                       choices=['bpe', 'unigram'],
                       help='Model type: bpe hoặc unigram')
    
    args = parser.parse_args()
    
    vi_tok, en_tok = train_tokenizers_from_csv(
        csv_path=args.csv_path,
        vi_column=args.vi_col,
        en_column=args.en_col,
        output_dir=args.output_dir,
        vi_vocab_size=args.vi_vocab_size,
        en_vocab_size=args.en_vocab_size,
        sample_size=args.sample_size,
        model_type=args.model_type
    )
    
    print("\n" + "="*70)
    print(" CÁCH SỬ DỤNG TRONG CODE CỦA BẠN:")
    print("="*70)
    print("""
# Load tokenizers
from tokenizer_sentencepiece import SentencePieceTokenizer

vi_tokenizer = SentencePieceTokenizer('../data/processed/vi_sp.model')
en_tokenizer = SentencePieceTokenizer('../data/processed/en_sp.model')

# Encode
text = "Xin chào"
ids = vi_tokenizer.encode(text)  # [2, 1234, 5678, 3]

# Decode
decoded = vi_tokenizer.decode(ids)  # "Xin chào"

# Vocab size
print(len(vi_tokenizer))  # 32000
""")