"""
GRADIO UI - BIDIRECTIONAL VIETNAMESE ↔ ENGLISH TRANSLATION
Instagram-inspired modern light theme interface
WITH SENTENCEPIECE TOKENIZERS - GRADIO 6.0 COMPATIBLE
"""

import gradio as gr
import torch
import pickle
import os
import glob
from pathlib import Path

# Import your modules - UPDATED FOR SENTENCEPIECE
from tokenizer_sentencepiece import SentencePieceTokenizer
from complete_transformer import create_model
from inference_evaluation_v2 import translate_sentence, beam_search_decode, greedy_decode

# ============================================================================
# LOAD MODELS & TOKENIZERS (UPDATED)
# ============================================================================

class BidirectionalTranslationModel:
    """Wrapper class để quản lý cả 2 models và SentencePiece tokenizers"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_vi_en = None
        self.model_en_vi = None
        self.vi_tokenizer = None
        self.en_tokenizer = None
        self.model_info_vi_en = {}
        self.model_info_en_vi = {}
        
    def load_tokenizers(self):
        """Load SentencePiece tokenizers - UPDATED"""
        print("📚 Loading SentencePiece tokenizers...")
        
        vi_model_path = '../data/processed/vi_sp.model'
        en_model_path = '../data/processed/en_sp.model'
        
        try:
            self.vi_tokenizer = SentencePieceTokenizer(vi_model_path)
            self.en_tokenizer = SentencePieceTokenizer(en_model_path)
            
            print(f"✓ Loaded tokenizers (VI: {len(self.vi_tokenizer)}, EN: {len(self.en_tokenizer)})")
            return True
        except Exception as e:
            print(f"✗ Không thể load tokenizers: {e}")
            return False
    
    def load_model(self, direction='vi-en'):
        """Load model cho một chiều dịch"""
        print(f"\n📄 Loading {direction.upper()} model...")
        
        if direction == 'vi-en':
            checkpoint_dir = 'checkpoints/vi_en'
            src_tokenizer = self.vi_tokenizer
            tgt_tokenizer = self.en_tokenizer
        else:
            checkpoint_dir = 'checkpoints/en_vi'
            src_tokenizer = self.en_tokenizer
            tgt_tokenizer = self.vi_tokenizer
        
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        
        if os.path.exists(best_model_path):
            checkpoint_path = best_model_path
            print(f"✓ Found best model: {best_model_path}")
        else:
            checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt'))
            if not checkpoints:
                latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
                if os.path.exists(latest_path):
                    checkpoint_path = latest_path
                    print(f"✓ Found latest checkpoint: {latest_path}")
                else:
                    print(f"✗ Không tìm thấy checkpoint nào trong {checkpoint_dir}")
                    return False
            else:
                checkpoint_path = max(checkpoints, key=os.path.getmtime)
                print(f"✓ Found latest checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            model_size = 'tiny'
            for size in ['tiny', 'small', 'base', 'large']:
                if size in checkpoint_path.lower():
                    model_size = size
                    break
            
            print(f"🔨 Creating {model_size} model...")
            model, model_config = create_model(
                src_vocab_size=len(src_tokenizer),
                tgt_vocab_size=len(tgt_tokenizer),
                model_size=model_size,
                pad_idx=0
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            model_info = {
                'checkpoint': checkpoint_path,
                'epoch': checkpoint.get('epoch', 'N/A'),
                'val_loss': checkpoint.get('val_loss', 'N/A'),
                'model_size': model_size,
                'device': str(self.device),
                'parameters': sum(p.numel() for p in model.parameters()),
                'direction': direction,
                'src_vocab_size': len(src_tokenizer),
                'tgt_vocab_size': len(tgt_tokenizer)
            }
            
            if direction == 'vi-en':
                self.model_vi_en = model
                self.model_info_vi_en = model_info
            else:
                self.model_en_vi = model
                self.model_info_en_vi = model_info
            
            print(f"✓ {direction.upper()} model loaded successfully!")
            print(f"  - Epoch: {model_info['epoch']}")
            print(f"  - Val Loss: {model_info['val_loss']}")
            print(f"  - Parameters: {model_info['parameters']:,}")
            
            return True
            
        except Exception as e:
            print(f"✗ Lỗi khi load {direction} model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_all_models(self):
        """Load cả 2 models"""
        print("="*70)
        print("🚀 LOADING BIDIRECTIONAL MODELS (SENTENCEPIECE)")
        print("="*70)
        
        if not self.load_tokenizers():
            return False
        
        success_vi_en = self.load_model('vi-en')
        success_en_vi = self.load_model('en-vi')
        
        print("\n" + "="*70)
        if success_vi_en and success_en_vi:
            print("✅ CẢ 2 MODELS ĐÃ ĐƯỢC LOAD THÀNH CÔNG!")
        elif success_vi_en:
            print("⚠️  CHỈ VI->EN MODEL ĐƯỢC LOAD")
        elif success_en_vi:
            print("⚠️  CHỈ EN->VI MODEL ĐƯỢC LOAD")
        else:
            print("❌ KHÔNG LOAD ĐƯỢC MODEL NÀO")
        print("="*70)
        
        return success_vi_en or success_en_vi
    
    def translate(self, text, direction='vi-en', use_beam_search=True, beam_size=5):
        """Dịch văn bản - UPDATED for SentencePiece"""
        
        if direction == 'vi-en':
            if not self.model_vi_en:
                return "❌ VI->EN model chưa được load. Vui lòng reload model."
            model = self.model_vi_en
            src_tokenizer = self.vi_tokenizer
            tgt_tokenizer = self.en_tokenizer
            src_lang = 'vi'
        else:
            if not self.model_en_vi:
                return "❌ EN->VI model chưa được load. Vui lòng reload model."
            model = self.model_en_vi
            src_tokenizer = self.en_tokenizer
            tgt_tokenizer = self.vi_tokenizer
            src_lang = 'en'
        
        try:
            text = text.strip().lower()
            
            if not text:
                return ""
            
            translation = translate_sentence(
                model=model,
                sentence=text,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                device=self.device,
                use_beam_search=use_beam_search,
                beam_size=beam_size,
                max_len=100
            )
            
            return translation
        
        except Exception as e:
            import traceback
            error_msg = f"❌ Lỗi: {str(e)}"
            return error_msg

# Khởi tạo translator
translator = BidirectionalTranslationModel()

# ============================================================================
# GRADIO INTERFACE FUNCTIONS
# ============================================================================

def translate_text(input_text, src_lang, use_beam_search, beam_size):
    """Main translation function"""
    if not input_text or not input_text.strip():
        return ""
    
    direction = "vi-en" if src_lang == "vi" else "en-vi"
    result = translator.translate(input_text, direction, use_beam_search, beam_size)
    return result

def swap_languages(input_text, output_text, src_lang):
    """Swap source and target languages"""
    new_src = "en" if src_lang == "vi" else "vi"
    return output_text, input_text, new_src

def get_placeholder_text(src_lang):
    """Get placeholder text based on source language"""
    if src_lang == "vi":
        return "Nhập văn bản tiếng Việt..."
    else:
        return "Enter English text..."

def update_placeholder(src_lang):
    """Update placeholder when language changes"""
    return gr.update(placeholder=get_placeholder_text(src_lang))

def update_target_lang_html(src):
    """Update target language display"""
    if src == "vi":
        return '<div class="target-lang">🇬🇧 English</div>'
    else:
        return '<div class="target-lang">🇻🇳 Tiếng Việt</div>'

# ============================================================================
# INSTAGRAM-STYLE CSS
# ============================================================================

INSTAGRAM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    background: linear-gradient(180deg, #fafafa 0%, #ffffff 100%) !important;
}

.header-container {
    text-align: center;
    padding: 40px 20px 30px;
    background: white;
    border-bottom: 1px solid #dbdbdb;
    margin-bottom: 30px;
}

.header-title {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(45deg, #f58529, #dd2a7b, #8134af, #515bd4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}

.header-subtitle {
    color: #8e8e8e;
    font-size: 15px;
    font-weight: 400;
}

.lang-selector {
    background: white;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #dbdbdb;
    margin-bottom: 20px;
}

.gradio-radio label {
    background: white !important;
    border: 1.5px solid #dbdbdb !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: all 0.2s ease !important;
}

.gradio-radio label:hover {
    border-color: #0095f6 !important;
    background: #f8f9fa !important;
}

.gradio-textbox textarea {
    border: 1px solid #dbdbdb !important;
    border-radius: 12px !important;
    padding: 16px !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    transition: all 0.2s ease !important;
}

.gradio-textbox textarea:focus {
    border-color: #0095f6 !important;
    box-shadow: 0 0 0 1px #0095f6 !important;
}

.gradio-textbox label {
    font-weight: 600 !important;
    color: #262626 !important;
    font-size: 14px !important;
    margin-bottom: 8px !important;
}

.gradio-button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}

.gradio-button.primary {
    background: linear-gradient(45deg, #f58529, #dd2a7b, #8134af) !important;
    color: white !important;
    border: none !important;
}

.gradio-button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(245, 133, 41, 0.3) !important;
}

.target-lang {
    text-align: center;
    padding: 12px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 8px;
    color: white;
    font-weight: 600;
    font-size: 15px;
}

.info-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 20px;
    color: white;
    margin-top: 20px;
}

.info-box h3 {
    color: white !important;
    font-weight: 600;
    margin-bottom: 12px;
}

.footer {
    text-align: center;
    padding: 30px 20px;
    color: #8e8e8e;
    font-size: 13px;
    border-top: 1px solid #dbdbdb;
    margin-top: 40px;
}
"""

# ============================================================================
# BUILD GRADIO APP - INSTAGRAM STYLE
# ============================================================================

def create_app():
    """Tạo Gradio app với giao diện Instagram hiện đại"""
    
    with gr.Blocks(title="Translate • Instagram Style") as app:
        
        # Instagram-inspired CSS
        gr.HTML(f"<style>{INSTAGRAM_CSS}</style>")
        
        # Header
        gr.HTML("""
        <div class="header-container">
            <div class="header-title">✨ Translate</div>
            <div class="header-subtitle">Vietnamese ↔ English • Powered by AI</div>
        </div>
        """)
        
        # Language selector card
        with gr.Row():
            with gr.Column(scale=5):
                src_lang = gr.Radio(
                    choices=[("🇻🇳 Tiếng Việt", "vi"), ("🇬🇧 English", "en")],
                    value="vi",
                    label="Source Language",
                    show_label=False,
                    container=False
                )
            
            with gr.Column(scale=1, min_width=60):
                swap_btn = gr.Button("⇄", size="sm")
            
            with gr.Column(scale=5):
                target_lang_display = gr.HTML('<div class="target-lang">🇬🇧 English</div>')
        
        # Main translation area
        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="📝 Source Text",
                    placeholder="Nhập văn bản tiếng Việt...",
                    lines=10,
                    max_lines=20,
                    show_label=True
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", size="sm", scale=1)
                    translate_btn = gr.Button(
                        "✨ Translate",
                        variant="primary",
                        size="lg",
                        scale=2
                    )
            
            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="🎯 Translation",
                    placeholder="Your translation will appear here...",
                    lines=10,
                    max_lines=20,
                    interactive=False,
                    show_label=True
                )
        
        # Advanced settings
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                use_beam_search = gr.Checkbox(
                    label="🔍 Beam Search (Higher Quality)",
                    value=True,
                    info="Disable for faster translation"
                )
                beam_size = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Beam Size",
                    info="Higher = Better quality (1-10)"
                )
        
        # Examples section
        gr.HTML('<div style="margin-top: 30px; margin-bottom: 16px;"><h3 style="font-size: 20px; font-weight: 600; color: #262626;">💡 Try These Examples</h3></div>')
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Vietnamese → English**")
                gr.Examples(
                    examples=[
                        ["Xin chào! Hôm nay trời đẹp quá."],
                        ["Tôi đang học máy dịch tự động."],
                        ["Công nghệ AI đang phát triển rất nhanh."],
                    ],
                    inputs=input_text
                )
            
            with gr.Column(scale=1):
                gr.Markdown("**English → Vietnamese**")
                gr.Examples(
                    examples=[
                        ["Hello! The weather is beautiful today."],
                        ["I am learning machine translation."],
                        ["AI technology is developing very fast."],
                    ],
                    inputs=input_text
                )
        
        # Info section
        with gr.Accordion("ℹ️ About", open=False):
            gr.HTML("""
            <div class="info-box">
                <h3>✨ Features</h3>
                <p>• 🔄 Bidirectional translation: Vietnamese ↔ English</p>
                <p>• 🎯 High-quality Beam Search algorithm</p>
                <p>• ⚡ Fast Greedy decoding option</p>
                <p>• 🚫 No unknown words with SentencePiece</p>
                <p>• 🤖 Powered by Transformer neural networks</p>
            </div>
            """)
            
            gr.Markdown("""
            ### 📖 How to Use
            
            1. Select your source language (Vietnamese or English)
            2. Type or paste your text
            3. Click "Translate" or press Enter
            4. Use the ⇄ button to swap languages
            
            ### 💡 Tips for Best Results
            
            - **Beam Size 5**: Balanced quality and speed
            - **Beam Size 1**: Fastest (Greedy mode)
            - **Beam Size 10**: Highest quality
            - Works best with sentences of 5-20 words
            
            ---
            
            **Built with PyTorch • Transformer • SentencePiece**
            """)
        
        # Footer
        gr.HTML("""
        <div class="footer">
            Made with ❤️ using Gradio • Transformer Architecture • SentencePiece Tokenization
        </div>
        """)
        
        # Event handlers
        src_lang.change(
            fn=update_target_lang_html,
            inputs=src_lang,
            outputs=target_lang_display
        )
        
        src_lang.change(
            fn=update_placeholder,
            inputs=src_lang,
            outputs=input_text
        )
        
        translate_btn.click(
            fn=translate_text,
            inputs=[input_text, src_lang, use_beam_search, beam_size],
            outputs=output_text
        )
        
        input_text.submit(
            fn=translate_text,
            inputs=[input_text, src_lang, use_beam_search, beam_size],
            outputs=output_text
        )
        
        swap_btn.click(
            fn=swap_languages,
            inputs=[input_text, output_text, src_lang],
            outputs=[input_text, output_text, src_lang]
        )
        
        clear_btn.click(
            fn=lambda: ("", ""),
            outputs=[input_text, output_text]
        )
    
    return app

# ============================================================================
# LAUNCH APP
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🚀 KHỞI ĐỘNG TRANSLATION WEB APP (INSTAGRAM STYLE)")
    print("="*70)
    
    try:
        translator.load_all_models()
    except Exception as e:
        print(f"⚠️ Warning: {e}")
        print("💡 Bạn có thể reload models sau trong giao diện web.")
    
    app = create_app()
    
    print("\n" + "="*70)
    print("🌐 LAUNCHING WEB APP...")
    print("="*70)
    print("\n🔗 Access URLs:")
    print("  - Local:   http://localhost:7860")
    print("  - Network: http://0.0.0.0:7860")
    print("\n💡 Tips:")
    print("  - Set share=True to get a public URL")
    print("  - Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )