"""
CHECK VOCAB COVERAGE - Kiểm tra xem có cần expand vocab không
"""

import sys
sys.path.append('..')
from tokenizer_sentencepiece import SentencePieceTokenizer
from collections import Counter
from tqdm import tqdm

def check_vocab_coverage(tokenizer_path, text_file, language='vi'):
    """
    Kiểm tra vocab coverage trên medical data
    
    Args:
        tokenizer_path: Path to .model file
        text_file: Path to medical text file
        language: 'vi' or 'en'
    
    Returns:
        coverage_stats: Dict with coverage info
    """
    
    print("="*70)
    print("🔍 CHECKING VOCAB COVERAGE")
    print("="*70)
    
    # Load tokenizer
    print(f"\n📚 Loading tokenizer: {tokenizer_path}")
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    print(f"   Vocab size: {len(tokenizer):,}")
    
    # Load texts
    print(f"\n📖 Loading texts: {text_file}")
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"   Total lines: {len(texts):,}")
    
    # Analyze
    total_tokens = 0
    unk_tokens = 0
    long_sequences = []  # Sequences with many subwords
    problematic_words = Counter()
    
    print("\n🔬 Analyzing...")
    for text in tqdm(texts):
        # Encode as pieces (subwords)
        pieces = tokenizer.encode_as_pieces(text)
        total_tokens += len(pieces)
        
        # Count UNK
        unk_count = pieces.count('<unk>')
        unk_tokens += unk_count
        
        # Find words tokenized into many subwords
        words = text.split()
        for word in words:
            word_pieces = tokenizer.encode_as_pieces(word)
            
            # If 1 word → >4 subwords → problematic
            if len(word_pieces) > 4:
                problematic_words[word] += 1
                long_sequences.append({
                    'word': word,
                    'pieces': word_pieces,
                    'num_pieces': len(word_pieces)
                })
    
    # Calculate coverage
    coverage = 1 - (unk_tokens / total_tokens) if total_tokens > 0 else 0
    
    # Results
    print("\n" + "="*70)
    print("📊 COVERAGE RESULTS")
    print("="*70)
    
    print(f"\n✅ Overall Coverage: {coverage*100:.2f}%")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   UNK tokens: {unk_tokens:,}")
    print(f"   Known tokens: {total_tokens - unk_tokens:,}")
    
    # Decision
    print(f"\n💡 RECOMMENDATION:")
    if coverage >= 0.95:
        print(f"   ✅ Coverage is GOOD (>95%)")
        print(f"   ✅ NO NEED to expand vocab")
        print(f"   ✅ Reuse tokenizer from Bài 1")
    elif coverage >= 0.90:
        print(f"   ⚠️  Coverage is ACCEPTABLE (90-95%)")
        print(f"   💭 Consider testing both approaches:")
        print(f"      1. Reuse tokenizer (simpler)")
        print(f"      2. Expand vocab (may improve slightly)")
    else:
        print(f"   ❌ Coverage is LOW (<90%)")
        print(f"   ❌ SHOULD expand vocab or retrain tokenizer")
        print(f"   💡 Medical domain has many specialized terms")
    
    # Problematic words
    print(f"\n🔍 PROBLEMATIC WORDS (tokenized into many subwords):")
    print(f"   Found {len(problematic_words)} unique words with >4 subwords")
    
    if problematic_words:
        print(f"\n   Top 20 most frequent:")
        for word, count in problematic_words.most_common(20):
            pieces = tokenizer.encode_as_pieces(word)
            print(f"   - '{word}' ({count}x) → {pieces}")
    
    # Examples of long sequences
    print(f"\n📝 EXAMPLES of long tokenization:")
    shown = 0
    for item in long_sequences:
        if shown >= 10:
            break
        if item['num_pieces'] >= 6:  # Very long
            print(f"   - '{item['word']}' → {item['pieces']}")
            shown += 1
    
    # Stats summary
    stats = {
        'coverage': coverage,
        'total_tokens': total_tokens,
        'unk_tokens': unk_tokens,
        'problematic_words_count': len(problematic_words),
        'top_problematic': dict(problematic_words.most_common(100)),
        'recommendation': 'reuse' if coverage >= 0.95 else ('test_both' if coverage >= 0.90 else 'expand')
    }
    
    return stats


def compare_coverage(vi_tokenizer_path, en_tokenizer_path, 
                    vi_text_file, en_text_file):
    """
    So sánh coverage cho cả VI và EN
    """
    
    print("\n" + "="*70)
    print("🔄 COMPARING VI & EN COVERAGE")
    print("="*70)
    
    # Check VI
    print("\n🇻🇳 VIETNAMESE:")
    vi_stats = check_vocab_coverage(vi_tokenizer_path, vi_text_file, 'vi')
    
    # Check EN
    print("\n🇬🇧 ENGLISH:")
    en_stats = check_vocab_coverage(en_tokenizer_path, en_text_file, 'en')
    
    # Compare
    print("\n" + "="*70)
    print("📊 COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'Language':<15} {'Coverage':<12} {'UNK Tokens':<15} {'Recommendation'}")
    print("-"*70)
    print(f"{'Vietnamese':<15} {vi_stats['coverage']*100:>6.2f}%     {vi_stats['unk_tokens']:>10,}     {vi_stats['recommendation']}")
    print(f"{'English':<15} {en_stats['coverage']*100:>6.2f}%     {en_stats['unk_tokens']:>10,}     {en_stats['recommendation']}")
    
    # Overall recommendation
    print(f"\n💡 OVERALL RECOMMENDATION:")
    
    if vi_stats['recommendation'] == 'reuse' and en_stats['recommendation'] == 'reuse':
        print(f"   ✅ Both coverages are good")
        print(f"   ✅ REUSE tokenizers from Bài 1")
        print(f"   ✅ This is the simplest and recommended approach")
    elif 'expand' in [vi_stats['recommendation'], en_stats['recommendation']]:
        print(f"   ⚠️  One or both languages have low coverage")
        print(f"   💡 Consider expanding vocab for better performance")
    else:
        print(f"   💭 Mixed results - test both approaches")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--vi_tokenizer', type=str, 
                       default='../data/processed/vi_sp.model')
    parser.add_argument('--en_tokenizer', type=str,
                       default='../data/processed/en_sp.model')
    parser.add_argument('--vi_text', type=str, required=True,
                       help='Medical VI text file')
    parser.add_argument('--en_text', type=str, required=True,
                       help='Medical EN text file')
    
    args = parser.parse_args()
    
    # Run comparison
    compare_coverage(
        args.vi_tokenizer,
        args.en_tokenizer,
        args.vi_text,
        args.en_text
    )