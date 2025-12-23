import os
import sys
import argparse
import json
import torch
import pickle
from pathlib import Path
from datetime import datetime

from plot_checkpoint_history import load_history, plot_all_in_one, export_history_json
from inference_evaluation_v2 import (
    translate_sentence, 
    evaluate_translations,
    print_evaluation_results
)
from complete_transformer import create_model
from tokenizer_sentencepiece import SentencePieceTokenizer

class PipelineConfig:
    """Configuration for pipeline"""
    def __init__(self):
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        self.data_dir = '../data/processed'
        self.checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')
        self.results_dir = 'results'
        
        self.model_size = 'tiny'
        
        self.vi_eval_file = None
        self.en_eval_file = None
        
        self.use_beam_search = True
        self.beam_size = 5
        self.max_len = 100
        self.test_sample_size = None
        
        self.use_gemini = False
        self.gemini_api_key = 'AIzaSyDqHNpCj6xpao90muWoPOSwIBTcNzVk6Is'
        self.gemini_sample_size = 50
        
        self.use_comet = False
        self.comet_model = "Unbabel/wmt22-comet-da"
        self.comet_batch_size = 8
        self.comet_gpus = 1

def analyze_checkpoints(config):
    """Analyze checkpoints for both directions: VI→EN and EN→VI"""
    print("\n" + "="*70)
    print("CHECKPOINT ANALYSIS FOR BOTH DIRECTIONS")
    print("="*70)
    
    output_dir = os.path.join(config.results_dir, 'checkpoint_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    directions = ['vi_en', 'en_vi']
    results = {}
    
    for direction in directions:
        print(f"\n{'='*70}")
        print(f"Analyzing: {direction.upper()}")
        print(f"{'='*70}")
        
        checkpoint_patterns = [
            os.path.join(config.checkpoint_dir, direction, f'best_model_{direction}_16.pt'),
            os.path.join(config.checkpoint_dir, direction, 'best_model.pt'),
            os.path.join(config.checkpoint_dir, direction, 'latest_checkpoint.pt'),
        ]
        
        checkpoint_path = None
        for path in checkpoint_patterns:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if not checkpoint_path:
            print(f"No checkpoint found for {direction}")
            print(f"   Searched in: {config.checkpoint_dir}/{direction}/")
            results[direction] = None
            continue
        
        try:
            print(f"\nLoading checkpoint: {checkpoint_path}")
            history = load_history(checkpoint_path)
            
            plot_path = os.path.join(output_dir, f'training_history_{direction}.png')
            plot_all_in_one(
                history,
                save_path=plot_path,
                title=f"Training History - {direction.upper()}"
            )
            
            json_path = os.path.join(output_dir, f'history_{direction}.json')
            export_history_json(history, json_path)
            
            best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
            best_val_loss = min(history['val_loss'])
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            
            summary_path = os.path.join(output_dir, f'summary_{direction}.txt')
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write(f"CHECKPOINT ANALYSIS - {direction.upper()}\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"Checkpoint: {checkpoint_path}\n\n")
                
                f.write(f"Training Progress:\n")
                f.write(f"  Total epochs: {len(history['train_loss'])}\n")
                f.write(f"  Best epoch: {best_epoch}\n")
                f.write(f"  Best val loss: {best_val_loss:.4f}\n")
                f.write(f"  Final train loss: {final_train_loss:.4f}\n")
                f.write(f"  Final val loss: {final_val_loss:.4f}\n")
            
            results[direction] = {
                'checkpoint_path': checkpoint_path,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss
            }
            
            print(f"\n✅ Analysis complete for {direction}!")
            print(f"   📊 Plot: {plot_path}")
            print(f"   📋 History: {json_path}")
            print(f"   Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
            
        except Exception as e:
            print(f"\nError analyzing {direction}: {e}")
            results[direction] = None
    
    print(f"\n{'='*70}")
    print("CHECKPOINT ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    for direction in directions:
        if results[direction]:
            print(f"\n{direction.upper()}:")
            print(f"  Found checkpoint")
            print(f"  Best epoch: {results[direction]['best_epoch']}")
            print(f"  Best val loss: {results[direction]['best_val_loss']:.4f}")
        else:
            print(f"\n{direction.upper()}:")
            print(f"  No checkpoint found")
    
    print(f"\n📂 Results saved to: {output_dir}")
    
    return results

def load_eval_data(config):
    """Load evaluation data"""
    print("\nLoading evaluation data...")
    
    vi_tokenizer = SentencePieceTokenizer(os.path.join(config.data_dir, 'vi_sp.model'))
    en_tokenizer = SentencePieceTokenizer(os.path.join(config.data_dir, 'en_sp.model'))
    
    if config.vi_eval_file and config.en_eval_file:
        print(f"   Using custom eval files:")
        print(f"   VI: {config.vi_eval_file}")
        print(f"   EN: {config.en_eval_file}")
        
        with open(config.vi_eval_file, 'r', encoding='utf-8') as f:
            vi_texts = [line.strip() for line in f if line.strip()]
        
        with open(config.en_eval_file, 'r', encoding='utf-8') as f:
            en_texts = [line.strip() for line in f if line.strip()]
        
        if len(vi_texts) != len(en_texts):
            raise ValueError(f"Mismatch: VI has {len(vi_texts)} lines, EN has {len(en_texts)} lines")
        
        print(f"    Loaded {len(vi_texts)} sentence pairs")
        
        return vi_texts, en_texts, vi_tokenizer, en_tokenizer
    
    else:
        print(f"   Using test set from: {config.data_dir}/processed_data.pkl")
        
        with open(os.path.join(config.data_dir, 'processed_data.pkl'), 'rb') as f:
            processed_data = pickle.load(f)
        
        vi_data = processed_data['test']['src']
        en_data = processed_data['test']['tgt']
        
        vi_texts = [vi_tokenizer.decode(ids, skip_special_tokens=True) for ids in vi_data]
        en_texts = [en_tokenizer.decode(ids, skip_special_tokens=True) for ids in en_data]
        
        print(f"    Loaded {len(vi_texts)} test samples")
        
        return vi_texts, en_texts, vi_tokenizer, en_tokenizer

def evaluate_direction(direction, checkpoint_path, vi_texts, en_texts, 
                       vi_tokenizer, en_tokenizer, config):
    """Evaluate BLEU/COMET for one translation direction"""
    print(f"\n{'='*70}")
    print(f"🎯 EVALUATION: {direction.upper()}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    
    output_dir = os.path.join(config.results_dir, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if direction == 'vi_en':
            src_texts = vi_texts
            ref_texts = en_texts
            src_tokenizer = vi_tokenizer
            tgt_tokenizer = en_tokenizer
            src_lang = "Vietnamese"
            tgt_lang = "English"
        else:
            src_texts = en_texts
            ref_texts = vi_texts
            src_tokenizer = en_tokenizer
            tgt_tokenizer = vi_tokenizer
            src_lang = "English"
            tgt_lang = "Vietnamese"
        
        print(f"\n   Direction: {src_lang} → {tgt_lang}")
        print(f"   Total samples: {len(src_texts):,}")
        
        print(f"\nLoading model...")
        
        model, _ = create_model(
            src_vocab_size=len(src_tokenizer),
            tgt_vocab_size=len(tgt_tokenizer),
            model_size=config.model_size,
            pad_idx=0
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"   Loaded checkpoint: {checkpoint_path}")
        print(f"   Model size: {config.model_size}")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        
        if config.test_sample_size and config.test_sample_size < len(src_texts):
            import random
            indices = random.sample(range(len(src_texts)), config.test_sample_size)
            src_texts_sample = [src_texts[i] for i in indices]
            ref_texts_sample = [ref_texts[i] for i in indices]
            print(f"\n     Sampling {config.test_sample_size}/{len(src_texts)} samples")
        else:
            src_texts_sample = src_texts
            ref_texts_sample = ref_texts
        
        print(f"\nTranslating {len(src_texts_sample):,} sentences...")
        print(f"   Method: {'Beam Search' if config.use_beam_search else 'Greedy'}")
        if config.use_beam_search:
            print(f"   Beam size: {config.beam_size}")
        
        hypotheses = []
        
        from tqdm import tqdm
        
        for src_text in tqdm(src_texts_sample, desc="Translating"):
            hyp_text = translate_sentence(
                model, src_text, src_tokenizer, tgt_tokenizer, device,
                use_beam_search=config.use_beam_search,
                beam_size=config.beam_size,
                max_len=config.max_len
            )
            hypotheses.append(hyp_text)
        
        print(f"\nCalculating evaluation scores...")
        
        results = evaluate_translations(
            sources=src_texts_sample,
            references=ref_texts_sample,
            hypotheses=hypotheses,
            use_gemini=config.use_gemini,
            gemini_api_key=config.gemini_api_key,
            use_sacrebleu=True,
            use_comet=config.use_comet,
            comet_model=config.comet_model,
            comet_batch_size=config.comet_batch_size,
            comet_gpus=config.comet_gpus,
            sample_size=config.gemini_sample_size if config.use_gemini else None
        )
        
        print_evaluation_results(results, detailed=True)
        
        results_path = os.path.join(output_dir, f'results_{direction}.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            
            results_copy = results.copy()
            if 'comet_metrics' in results_copy and results_copy['comet_metrics']:
                if 'individual_scores' in results_copy['comet_metrics']:
                    del results_copy['comet_metrics']['individual_scores']
            
            json.dump(results_copy, f, indent=2, ensure_ascii=False)
        
        print(f"\n   Saved results: {results_path}")
        
        samples_path = os.path.join(output_dir, f'sample_translations_{direction}.txt')
        with open(samples_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"SAMPLE TRANSLATIONS - {direction.upper()}\n")
            f.write(f"{src_lang} → {tgt_lang}\n")
            f.write("="*70 + "\n\n")
            
            for i in range(min(20, len(src_texts_sample))):
                f.write(f"Example {i+1}:\n")
                f.write(f"  Source: {src_texts_sample[i]}\n")
                f.write(f"  Reference: {ref_texts_sample[i]}\n")
                f.write(f"  Hypothesis: {hypotheses[i]}\n")
                f.write("\n")
        
        print(f"   Saved samples: {samples_path}")
        
        eval_summary = {
            'success': True,
            'bleu_score': results['bleu_metrics']['corpus_bleu'],
            'n_samples': len(src_texts_sample),
            'results': results
        }
        
        if 'comet_metrics' in results and results['comet_metrics']:
            eval_summary['comet_score'] = results['comet_metrics']['comet_score']
        
        return eval_summary
        
    except Exception as e:
        print(f"\nError in {direction} evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'bleu_score': None,
            'comet_score': None,
            'error': str(e)
        }

def evaluate_both_directions(config, checkpoint_results):
    """Evaluate BLEU/COMET for both directions"""
    print("\n" + "="*70)
    print("EVALUATION FOR BOTH DIRECTIONS")
    print("="*70)
    
    try:
        vi_texts, en_texts, vi_tokenizer, en_tokenizer = load_eval_data(config)
    except Exception as e:
        print(f"\nError loading eval data: {e}")
        return None
    
    results = {}
    
    if checkpoint_results.get('vi_en') and checkpoint_results['vi_en']['checkpoint_path']:
        results['vi_en'] = evaluate_direction(
            'vi_en',
            checkpoint_results['vi_en']['checkpoint_path'],
            vi_texts, en_texts,
            vi_tokenizer, en_tokenizer,
            config
        )
    else:
        print(f"\n Skipping VI→EN: No checkpoint found")
        results['vi_en'] = {'success': False, 'bleu_score': None, 'comet_score': None}
    
    if checkpoint_results.get('en_vi') and checkpoint_results['en_vi']['checkpoint_path']:
        results['en_vi'] = evaluate_direction(
            'en_vi',
            checkpoint_results['en_vi']['checkpoint_path'],
            vi_texts, en_texts,
            vi_tokenizer, en_tokenizer,
            config
        )
    else:
        print(f"\n Skipping EN→VI: No checkpoint found")
        results['en_vi'] = {'success': False, 'bleu_score': None, 'comet_score': None}
    
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    
    for direction in ['vi_en', 'en_vi']:
        result = results[direction]
        if result['success']:
            print(f"\n{direction.upper()}:")
            print(f"  BLEU Score: {result['bleu_score']:.2f}")
            if 'comet_score' in result and result['comet_score']:
                print(f"  COMET Score: {result['comet_score']:.4f}")
            print(f"  Samples: {result['n_samples']:,}")
        else:
            print(f"\n{direction.upper()}:")
            print(f"  Evaluation failed")
    
    return results

def run_pipeline(config, run_checkpoint=True, run_evaluation=True):
    """Run pipeline"""
    print("\n" + "="*70)
    print("CHECKPOINT & EVALUATION PIPELINE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data dir: {config.data_dir}")
    print(f"  Model: {config.model_size}")
    print(f"  Checkpoint dir: {config.checkpoint_dir}")
    print(f"  Results dir: {config.results_dir}")
    print(f"  Run checkpoint analysis: {run_checkpoint}")
    print(f"  Run evaluation: {run_evaluation}")
    
    if config.use_comet:
        print(f"  COMET enabled: {config.comet_model}")
    
    if config.vi_eval_file and config.en_eval_file:
        print(f"\n  Custom eval files:")
        print(f"    VI: {config.vi_eval_file}")
        print(f"    EN: {config.en_eval_file}")
    
    os.makedirs(config.results_dir, exist_ok=True)
    
    checkpoint_results = None
    eval_results = None
    
    if run_checkpoint:
        checkpoint_results = analyze_checkpoints(config)
    else:
        print("\n⏭Skipping checkpoint analysis")
        checkpoint_results = {}
        for direction in ['vi_en', 'en_vi']:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, direction, f'best_model_{direction}.pt'
            )
            if os.path.exists(checkpoint_path):
                checkpoint_results[direction] = {'checkpoint_path': checkpoint_path}
            else:
                checkpoint_results[direction] = None
    
    if run_evaluation:
        if checkpoint_results:
            eval_results = evaluate_both_directions(config, checkpoint_results)
        else:
            print("\n Cannot run evaluation: No checkpoints found")
    else:
        print("\n⏭Skipping evaluation")
    
    print("\n" + "="*70)
    print(" PIPELINE SUMMARY")
    print("="*70)
    
    if checkpoint_results:
        print("\n✅ Checkpoint Analysis:")
        for direction in ['vi_en', 'en_vi']:
            if checkpoint_results.get(direction):
                print(f"  {direction.upper()}: Found")
            else:
                print(f"  {direction.upper()}: Not found")
    
    if eval_results:
        print("\nEvaluation:")
        for direction in ['vi_en', 'en_vi']:
            result = eval_results.get(direction, {})
            if result.get('success'):
                print(f"  {direction.upper()}: BLEU={result['bleu_score']:.2f}", end='')
                if 'comet_score' in result and result['comet_score']:
                    print(f", COMET={result['comet_score']:.4f}")
                else:
                    print()
            else:
                print(f"  {direction.upper()}: Failed")
    
    print(f"\n All results saved to: {config.results_dir}")
    
    print("\n" + "="*70)
    print(" PIPELINE COMPLETE!")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(
        description='Checkpoint Analysis & Evaluation Pipeline with COMET',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both steps for both directions
  python complete_pipeline.py --all
  
  # Run with COMET evaluation
  python complete_pipeline.py --all --use-comet
  
  # Checkpoint analysis only
  python complete_pipeline.py --checkpoint-only
  
  # Evaluation only
  python complete_pipeline.py --eval-only
  
  # Use custom eval files
  python complete_pipeline.py --all --vi-file path/to/vi.txt --en-file path/to/en.txt
  
  # Evaluate with beam search
  python complete_pipeline.py --all --beam-search --beam-size 10

  # Sample test set
  python complete_pipeline.py --all --test-samples 1000
  
  # Full evaluation with COMET
  python complete_pipeline.py --all --use-comet --comet-gpus 1
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run both checkpoint analysis and evaluation')
    parser.add_argument('--checkpoint-only', action='store_true',
                       help='Checkpoint analysis only')
    parser.add_argument('--eval-only', action='store_true',
                       help='Evaluation only')
    
    parser.add_argument('--data-dir', type=str, default='../data/processed',
                       help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory')
    
    parser.add_argument('--vi-file', type=str, default=None,
                       help='Vietnamese text file for evaluation')
    parser.add_argument('--en-file', type=str, default=None,
                       help='English text file for evaluation')
    
    parser.add_argument('--beam-search', action='store_true',
                       help='Use beam search (default: greedy)')
    parser.add_argument('--beam-size', type=int, default=5,
                       help='Beam size')
    parser.add_argument('--test-samples', type=int, default=None,
                       help='Number of test samples (None = all)')
    parser.add_argument('--use-gemini', action='store_true',
                       help='Use Gemini evaluation (requires API key)')
    parser.add_argument('--gemini-samples', type=int, default=50,
                       help='Number of samples for Gemini')
    
    parser.add_argument('--use-comet', action='store_true',
                       help='Use COMET evaluation')
    parser.add_argument('--comet-model', type=str, default='Unbabel/wmt22-comet-da',
                       help='COMET model name')
    parser.add_argument('--comet-batch-size', type=int, default=8,
                       help='COMET batch size')
    parser.add_argument('--comet-gpus', type=int, default=1,
                       help='Number of GPUs for COMET (0 for CPU)')
    
    args = parser.parse_args()
    
    if args.all:
        run_checkpoint = True
        run_evaluation = True
    elif args.checkpoint_only:
        run_checkpoint = True
        run_evaluation = False
    elif args.eval_only:
        run_checkpoint = False
        run_evaluation = True
    else:
        run_checkpoint = True
        run_evaluation = True
    
    config = PipelineConfig()
    
    config.data_dir = args.data_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.results_dir = args.results_dir
    
    config.vi_eval_file = args.vi_file
    config.en_eval_file = args.en_file
    
    config.use_beam_search = args.beam_search
    config.beam_size = args.beam_size
    config.test_sample_size = args.test_samples
    config.use_gemini = args.use_gemini
    config.gemini_sample_size = args.gemini_samples
    
    config.use_comet = args.use_comet
    config.comet_model = args.comet_model
    config.comet_batch_size = args.comet_batch_size
    config.comet_gpus = args.comet_gpus
    
    if config.vi_eval_file and not config.en_eval_file:
        parser.error("--vi-file requires --en-file")
    if config.en_eval_file and not config.vi_eval_file:
        parser.error("--en-file requires --vi-file")
    
    if config.vi_eval_file:
        if not os.path.exists(config.vi_eval_file):
            parser.error(f"VI file not found: {config.vi_eval_file}")
        if not os.path.exists(config.en_eval_file):
            parser.error(f"EN file not found: {config.en_eval_file}")
    
    run_pipeline(config, run_checkpoint, run_evaluation)

if __name__ == "__main__":
    main()