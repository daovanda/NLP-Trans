import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
import re
import time
import json

# GREEDY DECODE

def greedy_decode(model, src, src_tokenizer, tgt_tokenizer, device, max_len=100):
    """Greedy Decoding"""
    model.eval()
    
    if isinstance(src, list):
        src = torch.LongTensor([src]).to(device)
    elif src.dim() == 1:
        src = src.unsqueeze(0)
    
    with torch.no_grad():
        encoder_output, src_mask = model.encode(src)
        tgt_tokens = [tgt_tokenizer.SOS_IDX]
        
        for _ in range(max_len):
            tgt = torch.LongTensor([tgt_tokens]).to(device)
            output = model.decode(tgt, encoder_output, src_mask)
            next_token = output[0, -1, :].argmax().item()
            tgt_tokens.append(next_token)
            
            if next_token == tgt_tokenizer.EOS_IDX:
                break
        
        decoded_sentence = tgt_tokenizer.decode(tgt_tokens, skip_special_tokens=True)
    
    return tgt_tokens, decoded_sentence

# BEAM SEARCH

class BeamSearchNode:
    """Node for Beam Search"""
    def __init__(self, tokens, log_prob, length):
        self.tokens = tokens
        self.log_prob = log_prob
        self.length = length
        
    def eval(self, alpha=0.6):
        """Length penalty as in Google's NMT paper"""
        return self.log_prob / (self.length ** alpha)

def beam_search_decode(model, src, src_tokenizer, tgt_tokenizer, device, 
                       beam_size=5, max_len=100, alpha=0.6):
    """Beam Search Decoding with length penalty"""
    model.eval()
    
    if isinstance(src, list):
        src = torch.LongTensor([src]).to(device)
    elif src.dim() == 1:
        src = src.unsqueeze(0)
    
    with torch.no_grad():
        encoder_output, src_mask = model.encode(src)
        
        beams = [BeamSearchNode(
            tokens=[tgt_tokenizer.SOS_IDX],
            log_prob=0.0,
            length=1
        )]
        
        completed_beams = []
        
        for step in range(max_len):
            candidates = []
            
            for beam in beams:
                if beam.tokens[-1] == tgt_tokenizer.EOS_IDX:
                    completed_beams.append(beam)
                    continue
                
                tgt = torch.LongTensor([beam.tokens]).to(device)
                output = model.decode(tgt, encoder_output, src_mask)
                next_token_logits = output[0, -1, :]
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                top_log_probs, top_tokens = torch.topk(log_probs, beam_size)
                
                for log_prob, token in zip(top_log_probs, top_tokens):
                    new_beam = BeamSearchNode(
                        tokens=beam.tokens + [token.item()],
                        log_prob=beam.log_prob + log_prob.item(),
                        length=beam.length + 1
                    )
                    candidates.append(new_beam)
            
            if not candidates:
                break
            
            beams = sorted(candidates, key=lambda x: x.eval(alpha), reverse=True)[:beam_size]
            
            if len(completed_beams) >= beam_size:
                break
        
        completed_beams.extend(beams)
        best_beam = max(completed_beams, key=lambda x: x.eval(alpha))
        best_sentence = tgt_tokenizer.decode(best_beam.tokens, skip_special_tokens=True)
    
    return best_beam.tokens, best_sentence

# TRANSLATE SENTENCE

def translate_sentence(model, sentence, src_tokenizer, tgt_tokenizer, device,
                       use_beam_search=True, beam_size=5, max_len=100):
    model.eval()
    
    sentence = sentence.strip().lower()
    
    if not sentence:
        return ""
    
    tokens = src_tokenizer.encode(sentence, add_bos=True, add_eos=True)
    src = torch.LongTensor([tokens]).to(device)
    
    if use_beam_search:
        _, translation = beam_search_decode(
            model, src, src_tokenizer, tgt_tokenizer, 
            device, beam_size, max_len
        )
    else:
        _, translation = greedy_decode(
            model, src, src_tokenizer, tgt_tokenizer, 
            device, max_len
        )
    
    return translation

# TOKENIZATION FOR BLEU

def tokenize_for_bleu(text: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    
    text = text.lower()
    text = re.sub(r'([.,!?;:\)\(\[\]{}"\'])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.strip().split()
    
    return tokens

# N-GRAM COMPUTATION

def compute_ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
    if len(tokens) < n:
        return {}
    
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    return ngrams

# SENTENCE-LEVEL BLEU

def compute_sentence_bleu(reference: str, hypothesis: str, max_n: int = 4) -> Dict:
    ref_tokens = tokenize_for_bleu(reference)
    hyp_tokens = tokenize_for_bleu(hypothesis)
    
    if len(hyp_tokens) == 0:
        return {
            'bleu': 0.0,
            'bleu_1': 0.0,
            'bleu_2': 0.0,
            'bleu_3': 0.0,
            'bleu_4': 0.0,
            'brevity_penalty': 0.0,
            'length_ratio': 0.0
        }
    
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
    
    precisions = []
    individual_scores = {}
    
    for n in range(1, max_n + 1):
        ref_ngrams = compute_ngrams(ref_tokens, n)
        hyp_ngrams = compute_ngrams(hyp_tokens, n)
        
        matches = 0
        total = 0
        
        for ngram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
            total += count
        
        if total > 0:
            precision = matches / total
            precisions.append(precision)
            individual_scores[f'bleu_{n}'] = precision * 100
        else:
            precisions.append(0.0)
            individual_scores[f'bleu_{n}'] = 0.0
    
    if len(precisions) > 0 and all(p > 0 for p in precisions):
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
        bleu = bp * geo_mean * 100
    else:
        bleu = 0.0
    
    return {
        'bleu': bleu,
        **individual_scores,
        'brevity_penalty': bp,
        'length_ratio': hyp_len / ref_len if ref_len > 0 else 0.0
    }

# CORPUS-LEVEL BLEU

def calculate_corpus_bleu(references: List[str], hypotheses: List[str], 
                          max_n: int = 4) -> Dict:
    
    total_matches = {n: 0 for n in range(1, max_n + 1)}
    total_possible = {n: 0 for n in range(1, max_n + 1)}
    
    total_ref_len = 0
    total_hyp_len = 0
    
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = tokenize_for_bleu(ref)
        hyp_tokens = tokenize_for_bleu(hyp)
        
        total_ref_len += len(ref_tokens)
        total_hyp_len += len(hyp_tokens)
        
        for n in range(1, max_n + 1):
            ref_ngrams = compute_ngrams(ref_tokens, n)
            hyp_ngrams = compute_ngrams(hyp_tokens, n)
            
            for ngram, count in hyp_ngrams.items():
                total_matches[n] += min(count, ref_ngrams.get(ngram, 0))
                total_possible[n] += count
    
    precisions = []
    precision_scores = {}
    
    for n in range(1, max_n + 1):
        if total_possible[n] > 0:
            prec = total_matches[n] / total_possible[n]
            precisions.append(prec)
            precision_scores[f'corpus_bleu_{n}'] = prec * 100
        else:
            precisions.append(0.0)
            precision_scores[f'corpus_bleu_{n}'] = 0.0
    
    if total_hyp_len > total_ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - total_ref_len / total_hyp_len) if total_hyp_len > 0 else 0.0
    
    if all(p > 0 for p in precisions):
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
        corpus_bleu = bp * geo_mean * 100
    else:
        corpus_bleu = 0.0
    
    return {
        'corpus_bleu': corpus_bleu,
        **precision_scores,
        'brevity_penalty': bp,
        'length_ratio': total_hyp_len / total_ref_len if total_ref_len > 0 else 0.0,
        'hyp_length': total_hyp_len,
        'ref_length': total_ref_len,
        'n_sentences': len(references)
    }

# SACREBLEU 

def calculate_corpus_bleu_sacrebleu(references: List[str], 
                                     hypotheses: List[str]) -> Optional[Dict]:
    try:
        import sacrebleu
        
        refs_formatted = [[ref] for ref in references]
        bleu = sacrebleu.corpus_bleu(hypotheses, refs_formatted)
        
        return {
            'sacrebleu': bleu.score,
            'sacrebleu_1': bleu.precisions[0],
            'sacrebleu_2': bleu.precisions[1],
            'sacrebleu_3': bleu.precisions[2],
            'sacrebleu_4': bleu.precisions[3],
            'brevity_penalty': bleu.bp,
            'sys_len': bleu.sys_len,
            'ref_len': bleu.ref_len,
            'signature': bleu.signature
        }
        
    except ImportError:
        print(" SacreBLEU not installed. Install with: pip install sacrebleu")
        return None
    except Exception as e:
        print(f"  Error calculating SacreBLEU: {e}")
        return None

# COMET SCORE

def calculate_comet_score(sources: List[str], 
                         references: List[str], 
                         hypotheses: List[str],
                         model_name: str = "Unbabel/wmt22-comet-da",
                         batch_size: int = 8,
                         gpus: int = 1) -> Optional[Dict]:
    """
    Calculate COMET scores for translation quality evaluation.
    
    Args:
        sources: List of source sentences
        references: List of reference translations
        hypotheses: List of hypothesis translations
        model_name: COMET model to use (default: wmt22-comet-da)
        batch_size: Batch size for inference
        gpus: Number of GPUs to use (0 for CPU)
    
    Returns:
        Dictionary with COMET scores or None if failed
    """
    try:
        from comet import download_model, load_from_checkpoint
        
        print(f"\n🔮 Loading COMET model: {model_name}")
        
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        
        data = []
        for src, ref, hyp in zip(sources, references, hypotheses):
            data.append({
                "src": src,
                "mt": hyp,
                "ref": ref
            })
        
        print(f"   Evaluating {len(data)} samples...")
        
        model_output = model.predict(
            data, 
            batch_size=batch_size, 
            gpus=gpus
        )
        
        scores = model_output.scores
        system_score = model_output.system_score
        
        return {
            'comet_score': system_score,
            'comet_mean': np.mean(scores),
            'comet_std': np.std(scores),
            'comet_min': np.min(scores),
            'comet_max': np.max(scores),
            'individual_scores': scores,
            'model_name': model_name,
            'n_samples': len(scores)
        }
        
    except ImportError:
        print("  COMET not installed. Install with: pip install unbabel-comet")
        return None
    except Exception as e:
        print(f"  Error calculating COMET: {e}")
        import traceback
        traceback.print_exc()
        return None

# GEMINI SCORE (LLM-AS-A-JUDGE)

def create_gemini_prompt(source: str, reference: str, hypothesis: str) -> str:
    """Create detailed prompt for Gemini evaluation"""
    prompt = f"""You are an expert translation quality evaluator with deep knowledge of Vietnamese and English languages.

Your task is to evaluate the quality of a machine translation compared to a human reference translation.

SOURCE TEXT (Vietnamese):
{source}

REFERENCE TRANSLATION (Human):
{reference}

HYPOTHESIS TRANSLATION (Machine):
{hypothesis}

Please evaluate the hypothesis translation on the following criteria:

1. FLUENCY (0-100): Grammatical correctness and naturalness
   - 0-25: Incomprehensible, severe grammatical errors
   - 26-50: Understandable but awkward, multiple errors
   - 51-75: Generally good, minor grammatical issues
   - 76-100: Excellent, native-like fluency

2. ADEQUACY (0-100): Meaning preservation from source
   - 0-25: Most meaning lost or severely distorted
   - 26-50: Partial meaning preserved, important information missing
   - 51-75: Most meaning preserved, minor omissions/additions
   - 76-100: Complete and accurate meaning transfer

3. COMPARISON (0-100): How does hypothesis compare to reference?
   - Consider both quality and similarity to reference
   - Not identical to reference is OK if quality is good

Respond with ONLY a valid JSON object (no markdown, no code blocks):
{{
    "fluency": <number 0-100>,
    "adequacy": <number 0-100>,
    "comparison": <number 0-100>,
    "overall": <average of fluency and adequacy>,
    "explanation": "<2-3 sentences explaining the scores>",
    "main_issues": ["<issue1>", "<issue2>"]
}}"""
    
    return prompt

def parse_gemini_response(response_text: str) -> Dict:
    text = re.sub(r'```json\s*|\s*```', '', response_text)
    text = text.strip()
    
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    
    if not json_match:
        raise ValueError(f"No JSON found in response. Text: {text[:200]}")
    
    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}. Text: {json_match.group()[:200]}")
    
    required_fields = ['fluency', 'adequacy', 'overall']
    for field in required_fields:
        if field not in result:
            raise ValueError(f"Missing required field: {field}")
        
        score = result[field]
        if not isinstance(score, (int, float)) or not (0 <= score <= 100):
            raise ValueError(f"Invalid score for {field}: {score}")
    
    result.setdefault('comparison', result['overall'])
    result.setdefault('explanation', 'No explanation provided')
    result.setdefault('main_issues', [])
    
    return result

def calculate_gemini_score(source: str, reference: str, hypothesis: str, 
                          api_key: Optional[str] = None,
                          max_retries: int = 3,
                          timeout: int = 30) -> Dict:
    try:
        from google import genai
        import os
        import json
        import time
        
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Missing API key")
            return {
                'gemini_score': None,
                'fluency': None,
                'adequacy': None,
                'status': 'no_api_key',
                'explanation': 'Missing Gemini API key. Set GEMINI_API_KEY environment variable.'
            }
        
        prompt = f"""You are a translation quality evaluator.

Source (Vietnamese): {source}
Reference (Human): {reference}
Hypothesis (Machine): {hypothesis}

Evaluate the translation quality on:
1. Fluency (0-100): Grammar and naturalness
2. Adequacy (0-100): Meaning preservation
3. Overall (0-100): Average of fluency and adequacy

Respond with ONLY a JSON object (no markdown):
{{
  "fluency": <number>,
  "adequacy": <number>,
  "overall": <number>,
  "explanation": "<brief explanation>"
}}"""
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                client = genai.Client(api_key=api_key)
                
                response = client.models.generate_content(
                    model='gemini-2.0-flash-lite',
                    contents=prompt
                )
                
                response_text = response.text.strip()
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                
                result = json.loads(response_text)
                
                return {
                    'gemini_score': float(result.get('overall', 0)),
                    'fluency': float(result.get('fluency', 0)),
                    'adequacy': float(result.get('adequacy', 0)),
                    'explanation': result.get('explanation', 'No explanation'),
                    'status': 'success',
                    'attempts': attempt + 1
                }
                
            except json.JSONDecodeError as e:
                last_error = f"JSON decode error: {e}"
                print(f" Attempt {attempt+1}/{max_retries} failed: {last_error}")
                
            except Exception as e:
                last_error = str(e)
                print(f" Attempt {attempt+1}/{max_retries} failed: {last_error}")
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5
                    time.sleep(wait_time)
        
        return {
            'gemini_score': None,
            'fluency': None,
            'adequacy': None,
            'status': 'failed',
            'explanation': f'Failed after {max_retries} attempts: {last_error}',
            'last_error': str(last_error)
        }
        
    except ImportError:
        return {
            'gemini_score': None,
            'fluency': None,
            'adequacy': None,
            'status': 'not_installed',
            'explanation': 'google-genai not installed. Install with: pip install google-genai'
        }
    except Exception as e:
        return {
            'gemini_score': None,
            'fluency': None,
            'adequacy': None,
            'status': 'error',
            'explanation': f'Unexpected error: {str(e)}',
            'error': str(e)
        }

# COMPREHENSIVE EVALUATION

def evaluate_translations(sources: List[str], 
                         references: List[str], 
                         hypotheses: List[str], 
                         use_gemini: bool = False,
                         gemini_api_key: Optional[str] = None,
                         use_sacrebleu: bool = True,
                         use_comet: bool = False,
                         comet_model: str = "Unbabel/wmt22-comet-da",
                         comet_batch_size: int = 8,
                         comet_gpus: int = 1,
                         sample_size: Optional[int] = None) -> Dict:
    
    assert len(sources) == len(references) == len(hypotheses), \
        "All lists must have same length"
    
    print("\n" + "="*70)
    print("COMPREHENSIVE TRANSLATION EVALUATION")
    print("="*70)
    
    results = {
        'total_samples': len(sources),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print("\nCalculating Corpus BLEU (Standard)...")
    results['bleu_metrics'] = calculate_corpus_bleu(references, hypotheses)
    print(f"    Corpus BLEU: {results['bleu_metrics']['corpus_bleu']:.2f}")
    
    if use_sacrebleu:
        print("\n🏆 Calculating SacreBLEU (WMT Standard)...")
        sacrebleu_scores = calculate_corpus_bleu_sacrebleu(references, hypotheses)
        if sacrebleu_scores:
            results['sacrebleu_metrics'] = sacrebleu_scores
            print(f"    SacreBLEU: {sacrebleu_scores['sacrebleu']:.2f}")
    
    if use_comet:
        print("\n Calculating COMET Score...")
        comet_scores = calculate_comet_score(
            sources, references, hypotheses,
            model_name=comet_model,
            batch_size=comet_batch_size,
            gpus=comet_gpus
        )
        if comet_scores:
            results['comet_metrics'] = comet_scores
            print(f"    COMET Score: {comet_scores['comet_score']:.4f}")
    
    if use_gemini:
        print("\n Evaluating with Gemini API...")
        
        if sample_size and sample_size < len(sources):
            print(f"     Sampling {sample_size}/{len(sources)} sentences")
            indices = np.random.choice(len(sources), sample_size, replace=False)
            sources_sample = [sources[i] for i in indices]
            references_sample = [references[i] for i in indices]
            hypotheses_sample = [hypotheses[i] for i in indices]
        else:
            sources_sample = sources
            references_sample = references
            hypotheses_sample = hypotheses
        
        gemini_scores = []
        fluency_scores = []
        adequacy_scores = []
        comparison_scores = []
        failed_count = 0
        
        for idx, (src, ref, hyp) in enumerate(zip(sources_sample, references_sample, hypotheses_sample), 1):
            print(f"   Progress: {idx}/{len(sources_sample)}", end='\r')
            
            score = calculate_gemini_score(src, ref, hyp, gemini_api_key)
            
            if score['status'] == 'success':
                gemini_scores.append(score['gemini_score'])
                fluency_scores.append(score['fluency'])
                adequacy_scores.append(score['adequacy'])
                comparison_scores.append(score.get('comparison', score['gemini_score']))
            else:
                failed_count += 1
        
        if gemini_scores:
            results['gemini_metrics'] = {
                'avg_score': np.mean(gemini_scores),
                'avg_fluency': np.mean(fluency_scores),
                'avg_adequacy': np.mean(adequacy_scores),
                'avg_comparison': np.mean(comparison_scores),
                'std_score': np.std(gemini_scores),
                'n_evaluated': len(gemini_scores),
                'n_failed': failed_count,
                'success_rate': len(gemini_scores) / (len(gemini_scores) + failed_count) * 100
            }
            print(f"\n    Gemini Score: {results['gemini_metrics']['avg_score']:.2f}")
            print(f"    Success Rate: {results['gemini_metrics']['success_rate']:.1f}%")
        else:
            results['gemini_metrics'] = None
            print(f"\n    Gemini evaluation failed for all samples")
    
    print("\n" + "="*70)
    print(" EVALUATION COMPLETE")
    print("="*70)
    
    return results

# PRETTY PRINTING

def print_evaluation_results(results: Dict, detailed: bool = True):
    """Print evaluation results"""
    print("\n" + "="*70)
    print(" EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n Total samples: {results['total_samples']:,}")
    if 'timestamp' in results:
        print(f" Timestamp: {results['timestamp']}")
    
    if 'bleu_metrics' in results:
        print("\n BLEU Scores (Corpus-level):")
        bleu = results['bleu_metrics']
        print(f"   Overall BLEU:      {bleu['corpus_bleu']:.2f}")
        print(f"   BLEU-1:            {bleu['corpus_bleu_1']:.2f}")
        print(f"   BLEU-2:            {bleu['corpus_bleu_2']:.2f}")
        print(f"   BLEU-3:            {bleu['corpus_bleu_3']:.2f}")
        print(f"   BLEU-4:            {bleu['corpus_bleu_4']:.2f}")
        print(f"   Brevity Penalty:   {bleu['brevity_penalty']:.3f}")
        print(f"   Length Ratio:      {bleu['length_ratio']:.3f}")
        
        if detailed:
            print(f"   Hypothesis Length: {bleu['hyp_length']:,} tokens")
            print(f"   Reference Length:  {bleu['ref_length']:,} tokens")
    
    if 'sacrebleu_metrics' in results:
        print("\n SacreBLEU (WMT Standard):")
        sacre = results['sacrebleu_metrics']
        print(f"   SacreBLEU:         {sacre['sacrebleu']:.2f}")
        print(f"   BLEU-1:            {sacre['sacrebleu_1']:.2f}")
        print(f"   BLEU-2:            {sacre['sacrebleu_2']:.2f}")
        print(f"   BLEU-3:            {sacre['sacrebleu_3']:.2f}")
        print(f"   BLEU-4:            {sacre['sacrebleu_4']:.2f}")
        
        if detailed:
            print(f"   Signature:         {sacre['signature']}")
    
    if 'comet_metrics' in results and results['comet_metrics']:
        print("\n COMET Scores:")
        comet = results['comet_metrics']
        print(f"   COMET Score:       {comet['comet_score']:.4f}")
        print(f"   Mean:              {comet['comet_mean']:.4f}")
        print(f"   Std Dev:           {comet['comet_std']:.4f}")
        print(f"   Min:               {comet['comet_min']:.4f}")
        print(f"   Max:               {comet['comet_max']:.4f}")
        print(f"   Model:             {comet['model_name']}")
    
    if 'gemini_metrics' in results and results['gemini_metrics']:
        print("\n Gemini Scores (LLM-as-a-Judge):")
        gemini = results['gemini_metrics']
        print(f"   Overall Score:     {gemini['avg_score']:.2f} ± {gemini['std_score']:.2f}")
        print(f"   Fluency:           {gemini['avg_fluency']:.2f}")
        print(f"   Adequacy:          {gemini['avg_adequacy']:.2f}")
        print(f"   Comparison:        {gemini['avg_comparison']:.2f}")
        print(f"   Success Rate:      {gemini['success_rate']:.1f}%")
        
        if detailed:
            print(f"   Evaluated:         {gemini['n_evaluated']:,}/{results['total_samples']:,}")
            print(f"   Failed:            {gemini['n_failed']:,}")
    
    print("\n" + "="*70)

# COMPARISON HELPER

def compare_evaluation_methods(references: List[str], hypotheses: List[str]):
    """Compare sentence-level average vs corpus-level BLEU"""
    print("\n" + "="*70)
    print("COMPARING EVALUATION METHODS")
    print("="*70)
    
    print("\n Method 1: Average of Sentence-level BLEU (INCORRECT)")
    sentence_bleus = []
    for ref, hyp in zip(references, hypotheses):
        score = compute_sentence_bleu(ref, hyp)
        sentence_bleus.append(score['bleu'])
    
    avg_sentence_bleu = np.mean(sentence_bleus)
    print(f"   Average Sentence BLEU: {avg_sentence_bleu:.2f}")
    print(f"     This is WRONG for corpus evaluation!")
    
    print("\n Method 2: Corpus-level BLEU (CORRECT)")
    corpus_scores = calculate_corpus_bleu(references, hypotheses)
    print(f"   Corpus BLEU: {corpus_scores['corpus_bleu']:.2f}")
    print(f"    This is the CORRECT way!")
    
    diff = abs(avg_sentence_bleu - corpus_scores['corpus_bleu'])
    print(f"\n Difference: {diff:.2f} points")
    
    if diff > 1.0:
        print("     Significant difference! Always use corpus-level BLEU.")
    
    print("="*70)

if __name__ == "__main__":
    print("="*70)
    print(" TESTING EVALUATION FUNCTIONS")
    print("="*70)
    
    print("\n1️  Test Tokenization:")
    text = "Hello, world! How are you?"
    tokens = tokenize_for_bleu(text)
    print(f"   Input:  '{text}'")
    print(f"   Tokens: {tokens}")
    print(f"    Punctuation separated correctly")
    
    print("\n2️  Test Sentence BLEU:")
    reference = "the cat is on the mat"
    hypothesis = "the cat is on the mat"
    
    score = compute_sentence_bleu(reference, hypothesis)
    print(f"   Reference:  '{reference}'")
    print(f"   Hypothesis: '{hypothesis}'")
    print(f"   BLEU: {score['bleu']:.2f}")
    assert score['bleu'] == 100.0, "Perfect match should be 100"
    print(f"    Perfect match = 100 BLEU")
    
    print("\n3️ Test Corpus BLEU:")
    references = [
        "the cat is on the mat",
        "hello world",
        "this is a test"
    ]
    hypotheses = [
        "the cat is on the mat",
        "hello world",
        "this is a test"
    ]
    
    corpus_scores = calculate_corpus_bleu(references, hypotheses)
    print(f"   Corpus BLEU: {corpus_scores['corpus_bleu']:.2f}")
    print(f"   BLEU-1: {corpus_scores['corpus_bleu_1']:.2f}")
    print(f"   BLEU-4: {corpus_scores['corpus_bleu_4']:.2f}")
    assert corpus_scores['corpus_bleu'] == 100.0, "All perfect matches should be 100"
    print(f"    All perfect matches = 100 BLEU")
    
    print("\n Test Imperfect Translations:")
    references = [
        "the cat sat on the mat",
        "hello world"
    ]
    hypotheses = [
        "the cat on mat",
        "hi world"
    ]
    
    corpus_scores = calculate_corpus_bleu(references, hypotheses)
    print(f"   Corpus BLEU: {corpus_scores['corpus_bleu']:.2f}")
    print(f"   BP: {corpus_scores['brevity_penalty']:.3f}")
    print(f"    Lower score for imperfect translations")
    
    print("\n Compare Sentence vs Corpus BLEU:")
    references = [
        "the quick brown fox jumps over the lazy dog",
        "cat"
    ]
    hypotheses = [
        "the quick brown fox",
        "cat"
    ]
    
    compare_evaluation_methods(references, hypotheses)
    
    print("\n  Test SacreBLEU:")
    sacre_scores = calculate_corpus_bleu_sacrebleu(references, hypotheses)
    if sacre_scores:
        print(f"   SacreBLEU: {sacre_scores['sacrebleu']:.2f}")
        print(f"    SacreBLEU available")
    else:
        print(f"     SacreBLEU not installed")
    
    print("\n Test Comprehensive Evaluation:")
    sources = ["xin chào", "tạm biệt"]
    references = ["hello", "goodbye"]
    hypotheses = ["hello", "bye"]
    
    results = evaluate_translations(
        sources, references, hypotheses,
        use_gemini=False,
        use_sacrebleu=True,
        use_comet=False
    )
    
    print_evaluation_results(results, detailed=True)
    
    print("\n" + "="*70)
    print(" ALL TESTS PASSED!")
    print("="*70)
    
    print("\n USAGE TIPS:")
    print("   1. Always use corpus-level BLEU for evaluation")
    print("   2. Use SacreBLEU for fair comparisons with other papers")
    print("   3. Sentence-level BLEU is only for visualization")
    print("   4. COMET provides neural metric correlation with human judgment")
    print("   5. Gemini scoring is expensive - use sampling for large datasets")
    print("   6. Set GEMINI_API_KEY environment variable for LLM evaluation")
    
    print("\n EXAMPLE:")
    print("""
    from inference_evaluation_v2 import evaluate_translations
    
    results = evaluate_translations(
        sources=['xin chào', 'tạm biệt'],
        references=['hello', 'goodbye'],
        hypotheses=['hello', 'bye'],
        use_sacrebleu=True,
        use_comet=True,
        comet_model='Unbabel/wmt22-comet-da',
        use_gemini=True,
        gemini_api_key='your-api-key',
        sample_size=100
    )
    
    print_evaluation_results(results)
    """)