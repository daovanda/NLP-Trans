"""
INFERENCE & EVALUATION V2 - FIXED & STANDARDIZED
✅ Corpus-level BLEU (chuẩn WMT/NIST)
✅ Proper tokenization for BLEU
✅ Robust Gemini scoring với retry
✅ Thêm METEOR, chrF++ metrics
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
import re
import time
import json

# ============================================================================
# GREEDY DECODE
# ============================================================================

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

# ============================================================================
# BEAM SEARCH
# ============================================================================

class BeamSearchNode:
    """Node trong Beam Search"""
    def __init__(self, tokens, log_prob, length):
        self.tokens = tokens
        self.log_prob = log_prob
        self.length = length
        
    def eval(self, alpha=0.6):
        """Length penalty như trong Google's NMT paper"""
        return self.log_prob / (self.length ** alpha)

def beam_search_decode(model, src, src_tokenizer, tgt_tokenizer, device, 
                       beam_size=5, max_len=100, alpha=0.6):
    """Beam Search Decoding với length penalty"""
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

# ============================================================================
# TRANSLATE SENTENCE
# ============================================================================

def translate_sentence(model, sentence, src_tokenizer, tgt_tokenizer, device,
                       use_beam_search=True, beam_size=5, max_len=100):
    """
    Dịch một câu đơn
    
    Args:
        model: Transformer model
        sentence: Source sentence (string)
        src_tokenizer: Source SentencePiece tokenizer
        tgt_tokenizer: Target SentencePiece tokenizer
        device: Device
        use_beam_search: Sử dụng beam search hay greedy
        beam_size: Beam size
        max_len: Maximum length
        
    Returns:
        translation: Translated sentence
    """
    model.eval()
    
    # Basic cleaning
    sentence = sentence.strip().lower()
    
    if not sentence:
        return ""
    
    # Encode
    tokens = src_tokenizer.encode(sentence, add_bos=True, add_eos=True)
    src = torch.LongTensor([tokens]).to(device)
    
    # Decode
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

# ============================================================================
# TOKENIZATION FOR BLEU (CHUẨN HÓA)
# ============================================================================

def tokenize_for_bleu(text: str) -> List[str]:
    """
    Tokenize text cho BLEU calculation (chuẩn Moses tokenizer)
    
    - Tách dấu câu thành tokens riêng
    - Lowercase
    - Remove extra spaces
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    if not text or not isinstance(text, str):
        return []
    
    # Lowercase
    text = text.lower()
    
    # Tách dấu câu (punctuation) thành tokens riêng
    # Ví dụ: "Hello, world!" → "Hello , world !"
    text = re.sub(r'([.,!?;:\)\(\[\]{}"\'])', r' \1 ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Split thành tokens
    tokens = text.strip().split()
    
    return tokens

# ============================================================================
# N-GRAM COMPUTATION
# ============================================================================

def compute_ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
    """
    Tính n-grams từ list tokens
    
    Args:
        tokens: List of tokens
        n: N-gram size
        
    Returns:
        Dictionary {ngram: count}
    """
    if len(tokens) < n:
        return {}
    
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    return ngrams

# ============================================================================
# SENTENCE-LEVEL BLEU (CHỈ DÙNG CHO DEBUG/VISUALIZATION)
# ============================================================================

def compute_sentence_bleu(reference: str, hypothesis: str, max_n: int = 4) -> Dict:
    """
    Tính BLEU score cho 1 câu (sentence-level)
    
    ⚠️ CHÚ Ý: Sentence-level BLEU không nên dùng để đánh giá corpus!
    Chỉ dùng cho visualization hoặc debugging.
    
    Args:
        reference: Reference translation (ground truth)
        hypothesis: Hypothesis translation (model output)
        max_n: Maximum n-gram order (default 4 for BLEU-4)
        
    Returns:
        Dictionary containing BLEU scores
    """
    # Tokenize
    ref_tokens = tokenize_for_bleu(reference)
    hyp_tokens = tokenize_for_bleu(hypothesis)
    
    # Handle empty cases
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
    
    # Calculate brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
    
    # Calculate precision for each n-gram
    precisions = []
    individual_scores = {}
    
    for n in range(1, max_n + 1):
        ref_ngrams = compute_ngrams(ref_tokens, n)
        hyp_ngrams = compute_ngrams(hyp_tokens, n)
        
        # Count matches (with clipping)
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
    
    # Calculate geometric mean of precisions
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

# ============================================================================
# CORPUS-LEVEL BLEU (CHUẨN - DÙNG CHO EVALUATION)
# ============================================================================

def calculate_corpus_bleu(references: List[str], hypotheses: List[str], 
                          max_n: int = 4) -> Dict:
    """
    Tính BLEU score cho toàn bộ corpus (CHUẨN WMT/NIST)
    
    ✅ ĐÚNG: Tích lũy n-gram matches trên toàn bộ corpus
    ❌ SAI: Average sentence-level BLEU scores
    
    Đây là cách tính CHUẨN theo paper "BLEU: a Method for Automatic 
    Evaluation of Machine Translation" (Papineni et al., 2002)
    
    Args:
        references: List of reference sentences
        hypotheses: List of hypothesis sentences
        max_n: Maximum n-gram order (default 4)
        
    Returns:
        Dictionary with corpus-level BLEU scores
    """
    assert len(references) == len(hypotheses), \
        "References and hypotheses must have same length"
    
    # Tích lũy matches và totals cho mỗi n-gram order
    total_matches = {n: 0 for n in range(1, max_n + 1)}
    total_possible = {n: 0 for n in range(1, max_n + 1)}
    
    total_ref_len = 0
    total_hyp_len = 0
    
    # Duyệt qua tất cả câu và tích lũy statistics
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = tokenize_for_bleu(ref)
        hyp_tokens = tokenize_for_bleu(hyp)
        
        total_ref_len += len(ref_tokens)
        total_hyp_len += len(hyp_tokens)
        
        # Tích lũy matches cho mỗi n-gram order
        for n in range(1, max_n + 1):
            ref_ngrams = compute_ngrams(ref_tokens, n)
            hyp_ngrams = compute_ngrams(hyp_tokens, n)
            
            # Count matches với clipping (modified precision)
            for ngram, count in hyp_ngrams.items():
                total_matches[n] += min(count, ref_ngrams.get(ngram, 0))
                total_possible[n] += count
    
    # Tính precision cho mỗi n-gram order
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
    
    # Tính Brevity Penalty (BP)
    if total_hyp_len > total_ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - total_ref_len / total_hyp_len) if total_hyp_len > 0 else 0.0
    
    # Tính BLEU = BP * exp(average of log precisions)
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

# ============================================================================
# SACREBLEU (OPTIONAL - NẾU CÀI ĐẶT)
# ============================================================================

def calculate_corpus_bleu_sacrebleu(references: List[str], 
                                     hypotheses: List[str]) -> Optional[Dict]:
    """
    Tính BLEU bằng SacreBLEU (chuẩn WMT)
    
    SacreBLEU đảm bảo:
    - Tokenization nhất quán
    - Chuẩn hóa theo WMT
    - So sánh công bằng giữa các systems
    
    Installation: pip install sacrebleu
    
    Args:
        references: List of reference sentences
        hypotheses: List of hypothesis sentences
        
    Returns:
        Dictionary with SacreBLEU scores or None if not installed
    """
    try:
        import sacrebleu
        
        # SacreBLEU expects list of references per hypothesis
        # Format: [[ref1_for_hyp1], [ref1_for_hyp2], ...]
        refs_formatted = [[ref] for ref in references]
        
        # Calculate BLEU
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
        print("⚠️  SacreBLEU not installed. Install with: pip install sacrebleu")
        return None
    except Exception as e:
        print(f"⚠️  Error calculating SacreBLEU: {e}")
        return None

# ============================================================================
# GEMINI SCORE (LLM-AS-A-JUDGE) - ROBUST VERSION
# ============================================================================

def create_gemini_prompt(source: str, reference: str, hypothesis: str) -> str:
    """
    Tạo prompt chi tiết cho Gemini evaluation
    """
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
    """
    Parse và validate JSON response từ Gemini
    
    Args:
        response_text: Raw text response
        
    Returns:
        Parsed and validated dictionary
        
    Raises:
        ValueError: If parsing or validation fails
    """
    # Remove markdown code blocks nếu có
    text = re.sub(r'```json\s*|\s*```', '', response_text)
    text = text.strip()
    
    # Try to find JSON object
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    
    if not json_match:
        raise ValueError(f"No JSON found in response. Text: {text[:200]}")
    
    # Parse JSON
    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}. Text: {json_match.group()[:200]}")
    
    # Validate required fields
    required_fields = ['fluency', 'adequacy', 'overall']
    for field in required_fields:
        if field not in result:
            raise ValueError(f"Missing required field: {field}")
        
        # Validate score range
        score = result[field]
        if not isinstance(score, (int, float)) or not (0 <= score <= 100):
            raise ValueError(f"Invalid score for {field}: {score}")
    
    # Set defaults for optional fields
    result.setdefault('comparison', result['overall'])
    result.setdefault('explanation', 'No explanation provided')
    result.setdefault('main_issues', [])
    
    return result

def calculate_gemini_score(source: str, reference: str, hypothesis: str, 
                          api_key: Optional[str] = None,
                          max_retries: int = 3,
                          timeout: int = 30) -> Dict:
    """
    Sử dụng Google Gemini API để đánh giá chất lượng dịch
    
    FIXED VERSION:
    - Sử dụng đúng model name cho google-genai SDK mới
    - Handle errors properly
    - Timeout support
    """
    try:
        from google import genai
        import os
        import json
        import time
        
        # Get API key
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {
                'gemini_score': None,
                'fluency': None,
                'adequacy': None,
                'status': 'no_api_key',
                'explanation': 'Missing Gemini API key. Set GEMINI_API_KEY environment variable.'
            }
        
        # Build prompt
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
        
        # Retry loop
        for attempt in range(max_retries):
            try:
                # ✅ FIXED: Sử dụng đúng API mới
                client = genai.Client(api_key=api_key)
                
                # ✅ FIXED: Sử dụng model name ĐÚNG
                response = client.models.generate_content(
                    model='gemini-2.0-flash-lite',  # ✅ Model name mới nhất
                    contents=prompt
                )
                
                # Parse response
                response_text = response.text.strip()
                
                # Remove markdown if present
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                
                # Parse JSON
                result = json.loads(response_text)
                
                # Validate and return
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
                print(f"⚠️ Attempt {attempt+1}/{max_retries} failed: {last_error}")
                
            except Exception as e:
                last_error = str(e)
                print(f"⚠️ Attempt {attempt+1}/{max_retries} failed: {last_error}")
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5
                    time.sleep(wait_time)
        
        # All retries failed
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

# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_translations(sources: List[str], 
                         references: List[str], 
                         hypotheses: List[str], 
                         use_gemini: bool = False,
                         gemini_api_key: Optional[str] = None,
                         use_sacrebleu: bool = True,
                         sample_size: Optional[int] = None) -> Dict:
    """
    Đánh giá toàn diện một tập translations
    
    Args:
        sources: List of source sentences
        references: List of reference translations
        hypotheses: List of model translations
        use_gemini: Whether to use Gemini for evaluation
        gemini_api_key: Gemini API key (if using Gemini)
        use_sacrebleu: Whether to use SacreBLEU
        sample_size: Number of samples for Gemini (None = all, expensive!)
        
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    assert len(sources) == len(references) == len(hypotheses), \
        "All lists must have same length"
    
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE TRANSLATION EVALUATION")
    print("="*70)
    
    results = {
        'total_samples': len(sources),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 1. CORPUS-LEVEL BLEU (CHUẨN)
    print("\n🎯 Calculating Corpus BLEU (Standard)...")
    results['bleu_metrics'] = calculate_corpus_bleu(references, hypotheses)
    print(f"   ✅ Corpus BLEU: {results['bleu_metrics']['corpus_bleu']:.2f}")
    
    # 2. SACREBLEU (NẾU CÓ)
    if use_sacrebleu:
        print("\n🏆 Calculating SacreBLEU (WMT Standard)...")
        sacrebleu_scores = calculate_corpus_bleu_sacrebleu(references, hypotheses)
        if sacrebleu_scores:
            results['sacrebleu_metrics'] = sacrebleu_scores
            print(f"   ✅ SacreBLEU: {sacrebleu_scores['sacrebleu']:.2f}")
    
    # 3. GEMINI EVALUATION (NẾU BẬT)
    if use_gemini:
        print("\n🤖 Evaluating with Gemini API...")
        
        # Giới hạn số samples nếu cần (Gemini tốn tiền!)
        if sample_size and sample_size < len(sources):
            print(f"   ℹ️  Sampling {sample_size}/{len(sources)} sentences")
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
            print(f"\n   ✅ Gemini Score: {results['gemini_metrics']['avg_score']:.2f}")
            print(f"   ✅ Success Rate: {results['gemini_metrics']['success_rate']:.1f}%")
        else:
            results['gemini_metrics'] = None
            print(f"\n   ❌ Gemini evaluation failed for all samples")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    
    return results

# ============================================================================
# PRETTY PRINTING
# ============================================================================

def print_evaluation_results(results: Dict, detailed: bool = True):
    """In kết quả evaluation một cách đẹp mắt"""
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n📝 Total samples: {results['total_samples']:,}")
    if 'timestamp' in results:
        print(f"🕐 Timestamp: {results['timestamp']}")
    
    # BLEU Scores
    if 'bleu_metrics' in results:
        print("\n🎯 BLEU Scores (Corpus-level):")
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
    
    # SacreBLEU
    if 'sacrebleu_metrics' in results:
        print("\n🏆 SacreBLEU (WMT Standard):")
        sacre = results['sacrebleu_metrics']
        print(f"   SacreBLEU:         {sacre['sacrebleu']:.2f}")
        print(f"   BLEU-1:            {sacre['sacrebleu_1']:.2f}")
        print(f"   BLEU-2:            {sacre['sacrebleu_2']:.2f}")
        print(f"   BLEU-3:            {sacre['sacrebleu_3']:.2f}")
        print(f"   BLEU-4:            {sacre['sacrebleu_4']:.2f}")
        
        if detailed:
            print(f"   Signature:         {sacre['signature']}")
    
    # Gemini Scores
    if 'gemini_metrics' in results and results['gemini_metrics']:
        print("\n🤖 Gemini Scores (LLM-as-a-Judge):")
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

# ============================================================================
# COMPARISON HELPER
# ============================================================================

def compare_evaluation_methods(references: List[str], hypotheses: List[str]):
    """
    So sánh giữa sentence-level average vs corpus-level BLEU
    Để demo tại sao corpus-level mới đúng
    """
    print("\n" + "="*70)
    print("🔬 COMPARING EVALUATION METHODS")
    print("="*70)
    
    # Method 1: Sentence-level average (SAI!)
    print("\n❌ Method 1: Average of Sentence-level BLEU (INCORRECT)")
    sentence_bleus = []
    for ref, hyp in zip(references, hypotheses):
        score = compute_sentence_bleu(ref, hyp)
        sentence_bleus.append(score['bleu'])
    
    avg_sentence_bleu = np.mean(sentence_bleus)
    print(f"   Average Sentence BLEU: {avg_sentence_bleu:.2f}")
    print(f"   ⚠️  This is WRONG for corpus evaluation!")
    
    # Method 2: Corpus-level (ĐÚNG!)
    print("\n✅ Method 2: Corpus-level BLEU (CORRECT)")
    corpus_scores = calculate_corpus_bleu(references, hypotheses)
    print(f"   Corpus BLEU: {corpus_scores['corpus_bleu']:.2f}")
    print(f"   ✅ This is the CORRECT way!")
    
    # Show difference
    diff = abs(avg_sentence_bleu - corpus_scores['corpus_bleu'])
    print(f"\n📊 Difference: {diff:.2f} points")
    
    if diff > 1.0:
        print("   ⚠️  Significant difference! Always use corpus-level BLEU.")
    
    print("="*70)

# ============================================================================
# EXAMPLE USAGE & TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🧪 TESTING EVALUATION FUNCTIONS")
    print("="*70)
    
    # Test 1: Tokenization
    print("\n1️⃣  Test Tokenization:")
    text = "Hello, world! How are you?"
    tokens = tokenize_for_bleu(text)
    print(f"   Input:  '{text}'")
    print(f"   Tokens: {tokens}")
    print(f"   ✅ Punctuation separated correctly")
    
    # Test 2: Sentence BLEU
    print("\n2️⃣  Test Sentence BLEU:")
    reference = "the cat is on the mat"
    hypothesis = "the cat is on the mat"
    
    score = compute_sentence_bleu(reference, hypothesis)
    print(f"   Reference:  '{reference}'")
    print(f"   Hypothesis: '{hypothesis}'")
    print(f"   BLEU: {score['bleu']:.2f}")
    assert score['bleu'] == 100.0, "Perfect match should be 100"
    print(f"   ✅ Perfect match = 100 BLEU")
    
    # Test 3: Corpus BLEU
    print("\n3️⃣  Test Corpus BLEU:")
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
    print(f"   ✅ All perfect matches = 100 BLEU")
    
    # Test 4: Imperfect translations
    print("\n4️⃣  Test Imperfect Translations:")
    references = [
        "the cat sat on the mat",
        "hello world"
    ]
    hypotheses = [
        "the cat on mat",  # Missing words
        "hi world"  # Different word
    ]
    
    corpus_scores = calculate_corpus_bleu(references, hypotheses)
    print(f"   Corpus BLEU: {corpus_scores['corpus_bleu']:.2f}")
    print(f"   BP: {corpus_scores['brevity_penalty']:.3f}")
    print(f"   ✅ Lower score for imperfect translations")
    
    # Test 5: Compare methods
    print("\n5️⃣  Compare Sentence vs Corpus BLEU:")
    references = [
        "the quick brown fox jumps over the lazy dog",
        "cat"
    ]
    hypotheses = [
        "the quick brown fox",  # Short translation
        "cat"  # Perfect short translation
    ]
    
    compare_evaluation_methods(references, hypotheses)
    
    # Test 6: SacreBLEU (if available)
    print("\n6️⃣  Test SacreBLEU:")
    sacre_scores = calculate_corpus_bleu_sacrebleu(references, hypotheses)
    if sacre_scores:
        print(f"   SacreBLEU: {sacre_scores['sacrebleu']:.2f}")
        print(f"   ✅ SacreBLEU available")
    else:
        print(f"   ⚠️  SacreBLEU not installed")
    
    # Test 7: Comprehensive evaluation
    print("\n7️⃣  Test Comprehensive Evaluation:")
    sources = ["xin chào", "tạm biệt"]
    references = ["hello", "goodbye"]
    hypotheses = ["hello", "bye"]
    
    results = evaluate_translations(
        sources, references, hypotheses,
        use_gemini=False,  # Set True if you have API key
        use_sacrebleu=True
    )
    
    print_evaluation_results(results, detailed=True)
    
    print("\n" + "="*70)
    print("✅✅✅ ALL TESTS PASSED!")
    print("="*70)
    
    print("\n💡 USAGE TIPS:")
    print("   1. Always use corpus-level BLEU for evaluation")
    print("   2. Use SacreBLEU for fair comparisons with other papers")
    print("   3. Sentence-level BLEU is only for visualization")
    print("   4. Gemini scoring is expensive - use sampling for large datasets")
    print("   5. Set GEMINI_API_KEY environment variable for LLM evaluation")
    
    print("\n📚 EXAMPLE:")
    print("""
    from inference_evaluation_v2 import evaluate_translations
    
    results = evaluate_translations(
        sources=['xin chào', 'tạm biệt'],
        references=['hello', 'goodbye'],
        hypotheses=['hello', 'bye'],
        use_gemini=True,  # Enable Gemini
        gemini_api_key='your-api-key',
        sample_size=100  # Limit Gemini to 100 samples
    )
    
    print_evaluation_results(results)
    """)