# Transformer Machine Translation: Vietnamese → English

Dự án xây dựng mô hình Transformer từ đầu (from scratch) cho bài toán dịch máy Tiếng Việt - Tiếng Anh.

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Cài đặt](#cài-đặt)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Kiến trúc](#kiến-trúc)
- [Kết quả](#kết-quả)
- [Cải tiến](#cải-tiến)

---

## 🎯 Tổng quan

### Mục tiêu
Xây dựng hoàn chỉnh mô hình Transformer từ các thành phần cơ bản để thực hiện dịch máy Vi→En, đạt điểm cao nhất.

### Dataset
- **IWSLT 2015 Vietnamese-English**
- Train: ~133K cặp câu
- Validation: ~1.5K cặp câu  
- Test: ~1.3K cặp câu

### Highlights
✅ **100% code from scratch** - Tất cả components được implement tay  
✅ **Label Smoothing** - Giảm overfitting  
✅ **Warmup Learning Rate Scheduler** - Ổn định training  
✅ **Beam Search Decoding** - Cải thiện chất lượng dịch  
✅ **Bucket Sampling** - Tối ưu tốc độ training  
✅ **Dynamic Padding** - Giảm computation overhead  

---

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- CUDA 11.0+ (nếu dùng GPU)
- RAM: 8GB+ 
- Disk: 5GB+

### Cài đặt thư viện

```bash
# Clone repository
git clone <your-repo>
cd transformer-vi-en

# Tạo virtual environment (khuyên dùng)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets tokenizers sentencepiece tqdm matplotlib
```

### Kiểm tra cài đặt

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

---

## 📁 Cấu trúc dự án

```
transformer-vi-en/
├── data_preprocessing.py          # Xử lý dữ liệu IWSLT
├── dataloader_module.py           # DataLoader với bucket sampling
├── transformer_components.py      # Attention, FFN, Position Encoding
├── transformer_encoder_decoder.py # Encoder & Decoder layers
├── complete_transformer.py        # Mô hình Transformer hoàn chỉnh
├── training_module.py             # Training loop, Loss, Optimizer
├── inference_evaluation.py        # Decoding, BLEU score
├── main_pipeline.py               # Script chính
├── README.md                      # Hướng dẫn này
├── checkpoints/                   # Lưu model checkpoints
├── results/                       # Lưu kết quả, translations
└── logs/                          # Training logs
```

---

## 💻 Hướng dẫn sử dụng

### Quick Start - Chạy toàn bộ pipeline

```bash
# Chạy tất cả: Data → Train → Evaluate
python main_pipeline.py --stage all --model_size base --epochs 20 --batch_size 32
```

### Chạy từng bước

#### Bước 1: Chuẩn bị dữ liệu

```bash
python main_pipeline.py --stage 1
```

Tạo ra:
- `cleaned_data.json` - Dữ liệu đã làm sạch
- `vi_vocab.pkl` - Vocabulary tiếng Việt
- `en_vocab.pkl` - Vocabulary tiếng Anh
- `processed_data.pkl` - Dữ liệu đã encode

#### Bước 2-3: Tạo model và DataLoaders

```bash
python main_pipeline.py --stage 3 --model_size base
```

**Model sizes:**
- `tiny`: 2 layers, 256 dim → ~5M params (test nhanh)
- `small`: 4 layers, 256 dim → ~10M params (laptop, CPU)
- `base`: 6 layers, 512 dim → ~65M params (khuyên dùng, cần GPU)
- `large`: 6 layers, 1024 dim → ~260M params (GPU mạnh)

#### Bước 4: Training

```bash
python main_pipeline.py --stage 4 --model_size base --epochs 20 --batch_size 32
```

**Hyperparameters quan trọng:**
- `--epochs`: Số epochs (khuyên 15-20)
- `--batch_size`: Batch size (32 cho GPU 8GB, 64 cho GPU 16GB+)
- `--model_size`: Kích thước model

**Monitoring training:**
- Loss & Perplexity hiển thị real-time
- Checkpoints tự động lưu mỗi epoch
- Best model lưu vào `checkpoints/best_model.pt`

#### Bước 5: Evaluation

```bash
python main_pipeline.py --stage 5 --beam_size 5
```

**Outputs:**
- BLEU scores (Greedy vs Beam Search)
- Sample translations
- Full translations: `results/translations_beam.txt`
- Scores: `results/bleu_scores.json`

#### Bước 6: Interactive Translation

```bash
python main_pipeline.py --stage 6
```

Dịch câu tương tác:
```
Tiếng Việt: Xin chào, tôi là một sinh viên.
Tiếng Anh:  hello , i am a student .
```

---

## 🏗️ Kiến trúc

### Transformer Components

#### 1. Scaled Dot-Product Attention
```python
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

#### 2. Multi-Head Attention
- Parallel attention với nhiều heads (8 heads)
- Mỗi head học representation khác nhau
- Concat và project về d_model

#### 3. Positional Encoding
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

#### 4. Encoder Layer
```
Input 
→ Multi-Head Self-Attention 
→ Add & Norm 
→ Feed-Forward 
→ Add & Norm 
→ Output
```

#### 5. Decoder Layer
```
Input
→ Masked Self-Attention
→ Add & Norm
→ Cross-Attention (với Encoder)
→ Add & Norm
→ Feed-Forward
→ Add & Norm
→ Output
```

### Training Details

**Loss Function:**
- Label Smoothing Cross-Entropy
- Smoothing = 0.1 (giảm overconfidence)

**Optimizer:**
- Adam: β1=0.9, β2=0.98, ε=1e-9

**Learning Rate Schedule:**
```python
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
```
- Warmup: 4000 steps
- Tăng dần rồi giảm dần

**Regularization:**
- Dropout: 0.1
- Gradient Clipping: max_norm=1.0
- Label Smoothing: 0.1

### Decoding Strategies

#### Greedy Search
- Chọn token có xác suất cao nhất
- Nhanh nhưng không tối ưu

#### Beam Search
- Giữ top-K candidates
- Beam size = 5
- Length penalty: α = 0.6
- Chất lượng tốt hơn ~2-3 BLEU

---

## 📊 Kết quả

### Expected Results (Base model, 20 epochs)

| Metric | Greedy Search | Beam Search (k=5) |
|--------|--------------|-------------------|
| BLEU Score | ~25-28 | ~27-30 |
| Inference Speed | Fast | Medium |

### Sample Translations

**Ví dụ 1:**
```
VI: tôi đang học về trí tuệ nhân tạo .
EN: i am learning about artificial intelligence .
```

**Ví dụ 2:**
```
VI: hôm nay thời tiết rất đẹp .
EN: the weather is very nice today .
```

### Training Curves

Sau training, kiểm tra:
- `results/training_history.png` - Đồ thị Loss, Perplexity, LR
- Training loss giảm ổn định
- Validation loss không tăng (không overfit)

---

## 🚀 Cải tiến để tăng điểm

### 1. Data Augmentation (Khuyên dùng ⭐)

**Back-translation:**
```python
# Dịch EN→VI rồi dùng làm training data
# Tăng ~2-3 BLEU
```

**Thêm dữ liệu:**
```python
# Thêm TED Talks, OpenSubtitles
# Tăng vocabulary coverage
```

### 2. Model Improvements

**Relative Positional Encoding:**
- Thay Sinusoidal bằng Relative
- Tốt hơn cho câu dài

**Layer Normalization Position:**
```python
# Pre-LN thay vì Post-LN
# Ổn định hơn, train sâu hơn được
x = x + sublayer(norm(x))  # Pre-LN
```

**Tied Embeddings:**
```python
# Share weights giữa encoder embedding và decoder embedding
# Giảm parameters ~10%
```

### 3. Training Tricks (Dễ implement ⭐)

**Gradient Accumulation:**
```python
# Tăng effective batch size
# Không cần GPU lớn
accumulation_steps = 4
```

**Mixed Precision Training:**
```python
# Dùng float16 thay float32
# Nhanh gấp 2x, ít VRAM hơn
from torch.cuda.amp import autocast, GradScaler
```

**Longer Training:**
```python
# Train 30-40 epochs thay vì 20
# Tăng ~1-2 BLEU
```

### 4. Hyperparameter Tuning

**Tăng model size:**
```python
# Base → Large
# +5-8 BLEU nhưng cần GPU mạnh
```

**Tăng beam size:**
```python
# beam_size = 10
# +0.5-1 BLEU
```

**Label smoothing:**
```python
# Thử 0.05, 0.1, 0.2
# Tìm optimal value
```

### 5. Ensemble (Tăng nhiều nhất ⭐⭐⭐)

```python
# Train 3-5 models với random seeds khác nhau
# Average predictions
# +3-5 BLEU
```

### 6. Post-processing

**Moses Detokenizer:**
- Cải thiện quality
- Xử lý dấu câu đúng hơn

**Unknown word replacement:**
- Copy từ source sang target nếu UNK

---

## 🐛 Troubleshooting

### Out of Memory

```python
# Giảm batch_size
--batch_size 16

# Hoặc dùng gradient accumulation
accumulation_steps = 2
```

### Training quá chậm

```python
# Dùng model nhỏ hơn
--model_size small

# Tăng num_workers
num_workers = 4

# Dùng bucket sampling (đã có)
```

### BLEU score thấp

```python
# Train lâu hơn
--epochs 30

# Tăng model size
--model_size large

# Dùng beam search với beam_size lớn
--beam_size 10

# Data augmentation (back-translation)
```

### Overfitting

```python
# Tăng dropout
dropout = 0.2

# Tăng label smoothing
label_smoothing = 0.2

# Thêm dữ liệu
```

---

## 📚 References

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original Transformer paper

2. **The Annotated Transformer** (Harvard NLP)
   - http://nlp.seas.harvard.edu/annotated-transformer/

3. **Hugging Face Transformers**
   - https://huggingface.co/docs/transformers/

---

## 🎓 Báo cáo

### Nội dung báo cáo cần có:

#### 1. Xử lý dữ liệu
- Thống kê dataset (số câu, độ dài trung bình)
- Quy trình làm sạch
- Vocabulary size
- Ví dụ data sau preprocessing

#### 2. Kiến trúc
- Sơ đồ kiến trúc Transformer
- Chi tiết từng component (Attention, FFN, etc.)
- Số lượng parameters
- Hyperparameters đã chọn

#### 3. Training
- Đồ thị Loss/Perplexity (train & val)
- Đồ thị Learning Rate
- Training time
- Hardware specs

#### 4. Kết quả
- **BLEU scores** (Greedy vs Beam)
- Sample translations (10-20 ví dụ tốt)
- Analysis: Loại câu dịch tốt/kém

#### 5. So sánh cải tiến
- Baseline (no improvements)
- Với improvements (label smoothing, beam search, etc.)
- Bảng so sánh BLEU scores
- Ablation study nếu có

#### 6. Gemini Score (nếu yêu cầu)
- Dùng Gemini API để score translations
- So sánh với BLEU

---

## 📝 License

MIT License - Tự do sử dụng cho học tập

---

## 👨‍💻 Author

Dự án cho BTL NLP - Transformer Machine Translation

---

## 🙏 Acknowledgments

- IWSLT dataset
- PyTorch team
- Hugging Face Datasets

---

**Good luck với BTL! 🚀**

Nếu có vấn đề, mở issue hoặc liên hệ qua email.