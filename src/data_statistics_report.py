"""
DATA STATISTICS REPORT - Phần A của đồ án
Tạo báo cáo chi tiết về dữ liệu sau preprocessing
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm


def to_python_types(obj):
    """Convert numpy types to native Python types recursively."""
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
def analyze_sequence_lengths(processed_data, split='train'):
    """Phân tích độ dài sequences"""
    src_lengths = [len(seq) for seq in processed_data[split]['src']]
    tgt_lengths = [len(seq) for seq in processed_data[split]['tgt']]
    
    stats = {
        'src': {
            'mean': np.mean(src_lengths),
            'std': np.std(src_lengths),
            'min': np.min(src_lengths),
            'max': np.max(src_lengths),
            'median': np.median(src_lengths),
            'percentiles': {
                '25': np.percentile(src_lengths, 25),
                '50': np.percentile(src_lengths, 50),
                '75': np.percentile(src_lengths, 75),
                '90': np.percentile(src_lengths, 90),
                '95': np.percentile(src_lengths, 95),
                '99': np.percentile(src_lengths, 99)
            }
        },
        'tgt': {
            'mean': np.mean(tgt_lengths),
            'std': np.std(tgt_lengths),
            'min': np.min(tgt_lengths),
            'max': np.max(tgt_lengths),
            'median': np.median(tgt_lengths),
            'percentiles': {
                '25': np.percentile(tgt_lengths, 25),
                '50': np.percentile(tgt_lengths, 50),
                '75': np.percentile(tgt_lengths, 75),
                '90': np.percentile(tgt_lengths, 90),
                '95': np.percentile(tgt_lengths, 95),
                '99': np.percentile(tgt_lengths, 99)
            }
        }
    }
    
    return stats, src_lengths, tgt_lengths



def analyze_vocabulary(tokenizer, processed_data, split='train', lang='vi'):
    """Phân tích vocabulary usage"""
    # Đếm token frequency
    token_counts = Counter()
    
    for seq in tqdm(processed_data[split]['src' if lang == 'vi' else 'tgt'], 
                    desc=f"Analyzing {lang} vocab"):
        token_counts.update(seq)
    
    # Remove special tokens
    special_tokens = [tokenizer.PAD_IDX, tokenizer.UNK_IDX, 
                     tokenizer.SOS_IDX, tokenizer.EOS_IDX]
    for token in special_tokens:
        if token in token_counts:
            del token_counts[token]
    
    total_tokens = sum(token_counts.values())
    unique_tokens = len(token_counts)
    
    # Top tokens
    top_20 = token_counts.most_common(20)
    
    # Coverage analysis
    sorted_counts = sorted(token_counts.values(), reverse=True)
    cumsum = np.cumsum(sorted_counts)
    coverage_90 = np.searchsorted(cumsum, 0.9 * total_tokens) + 1
    coverage_95 = np.searchsorted(cumsum, 0.95 * total_tokens) + 1
    coverage_99 = np.searchsorted(cumsum, 0.99 * total_tokens) + 1
    
    stats = {
        'vocab_size': len(tokenizer),
        'unique_tokens_used': unique_tokens,
        'total_tokens': total_tokens,
        'coverage': {
            '90%': coverage_90,
            '95%': coverage_95,
            '99%': coverage_99
        },
        'top_20_tokens': [
            {
                'token': tokenizer.sp.id_to_piece(token_id),
                'count': count,
                'percentage': count / total_tokens * 100
            }
            for token_id, count in top_20
        ]
    }
    
    return stats



def plot_length_distribution(src_lengths, tgt_lengths, split='train', save_path='length_dist.png'):
    """Vẽ phân bố độ dài"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Source histogram
    axes[0, 0].hist(src_lengths, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Vietnamese Length Distribution ({split})')
    axes[0, 0].axvline(np.mean(src_lengths), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(src_lengths):.1f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Target histogram
    axes[0, 1].hist(tgt_lengths, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Sequence Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'English Length Distribution ({split})')
    axes[0, 1].axvline(np.mean(tgt_lengths), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(tgt_lengths):.1f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot comparison
    axes[1, 0].boxplot([src_lengths, tgt_lengths], labels=['Vietnamese', 'English'])
    axes[1, 0].set_ylabel('Sequence Length')
    axes[1, 0].set_title('Length Comparison (Box Plot)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot
    if len(src_lengths) == len(tgt_lengths):
        sample_size = min(5000, len(src_lengths))
        indices = np.random.choice(len(src_lengths), sample_size, replace=False)
        axes[1, 1].scatter([src_lengths[i] for i in indices], 
                          [tgt_lengths[i] for i in indices],
                          alpha=0.3, s=1)
        axes[1, 1].set_xlabel('Vietnamese Length')
        axes[1, 1].set_ylabel('English Length')
        axes[1, 1].set_title(f'Length Correlation (sample={sample_size})')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot: {save_path}")
    plt.close()

def generate_data_report(data_dir='../data/processed', output_dir='../results'):
    """
    Tạo báo cáo hoàn chỉnh về dữ liệu
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("📊 GENERATING DATA STATISTICS REPORT")
    print("="*70)
    
    # Load data
    print("\n1️⃣ Loading data...")
    with open(f'{data_dir}/processed_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    # Load tokenizers
    from tokenizer_sentencepiece import SentencePieceTokenizer
    vi_tokenizer = SentencePieceTokenizer(f'{data_dir}/vi_sp.model')
    en_tokenizer = SentencePieceTokenizer(f'{data_dir}/en_sp.model')
    
    report = {
        'dataset_overview': {},
        'splits': {}
    }
    
    # Dataset overview
    print("\n2️⃣ Dataset Overview...")
    report['dataset_overview'] = {
        'total_samples': sum(len(processed_data[s]['src']) for s in ['train', 'validation', 'test']),
        'train_samples': len(processed_data['train']['src']),
        'val_samples': len(processed_data['validation']['src']),
        'test_samples': len(processed_data['test']['src']),
        'vi_vocab_size': len(vi_tokenizer),
        'en_vocab_size': len(en_tokenizer)
    }
    
    # Analyze each split
    for split in ['train', 'validation', 'test']:
        print(f"\n3️⃣ Analyzing {split} split...")
        
        # Length statistics
        length_stats, src_lengths, tgt_lengths = analyze_sequence_lengths(
            processed_data, split
        )
        
        # Plot distributions
        plot_path = f'{output_dir}/length_distribution_{split}.png'
        plot_length_distribution(src_lengths, tgt_lengths, split, plot_path)
        
        report['splits'][split] = {
            'n_samples': len(processed_data[split]['src']),
            'length_statistics': length_stats
        }
    
    # Vocabulary analysis (only on train)
    print("\n4️⃣ Analyzing vocabularies...")
    vi_vocab_stats = analyze_vocabulary(vi_tokenizer, processed_data, 'train', 'vi')
    en_vocab_stats = analyze_vocabulary(en_tokenizer, processed_data, 'train', 'en')
    
    report['vocabulary'] = {
        'vietnamese': vi_vocab_stats,
        'english': en_vocab_stats
    }
    
    # Save report
    report_path = f'{output_dir}/data_statistics_report.json'

    report_clean = to_python_types(report)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_clean, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved report: {report_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("📋 DATA STATISTICS SUMMARY")
    print("="*70)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Total samples: {report['dataset_overview']['total_samples']:,}")
    print(f"  Train: {report['dataset_overview']['train_samples']:,}")
    print(f"  Validation: {report['dataset_overview']['val_samples']:,}")
    print(f"  Test: {report['dataset_overview']['test_samples']:,}")
    
    print(f"\n📚 Vocabulary:")
    print(f"  Vietnamese: {report['dataset_overview']['vi_vocab_size']:,} tokens")
    print(f"  English: {report['dataset_overview']['en_vocab_size']:,} tokens")
    
    print(f"\n📏 Sequence Lengths (Train):")
    train_stats = report['splits']['train']['length_statistics']
    print(f"  Vietnamese: {train_stats['src']['mean']:.1f} ± {train_stats['src']['std']:.1f}")
    print(f"  English: {train_stats['tgt']['mean']:.1f} ± {train_stats['tgt']['std']:.1f}")
    
    print(f"\n💡 Top 5 Vietnamese tokens:")
    for i, token_info in enumerate(vi_vocab_stats['top_20_tokens'][:10], 1):
        print(f"  {i}. '{token_info['token']}': {token_info['count']:,} ({token_info['percentage']:.2f}%)")
    
    print(f"\n💡 Top 5 English tokens:")
    for i, token_info in enumerate(en_vocab_stats['top_20_tokens'][:10], 1):
        print(f"  {i}. '{token_info['token']}': {token_info['count']:,} ({token_info['percentage']:.2f}%)")
    
    print(f"\n🎯 Vocabulary Coverage (Train):")
    print(f"  VI: Top {vi_vocab_stats['coverage']['90%']:,} tokens = 90% coverage")
    print(f"  EN: Top {en_vocab_stats['coverage']['90%']:,} tokens = 90% coverage")
    
    print("\n" + "="*70)
    print("✅✅✅ REPORT GENERATION COMPLETE!")
    print("="*70)
    
    return report

if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '../data/processed'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '../results'
    
    report = generate_data_report(data_dir, output_dir)