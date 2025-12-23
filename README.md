Phần 1 -  xây dựng mô hình Transformer từ đầu (Nhóm 3 -  Lớp  INT3406 1)

Các file model, vocal nằm ở data/processed

Để chạy (tính điểm BLEU và COMMET) cần tải checkpoint từ "https://drive.google.com/drive/u/2/folders/1kp4_GocTuke2wuYqyuJXSa_4uukLj_r_" rồi để vào src/checkpoints/en-vi (vi-en) tương ứng

Sau đó chạy (cd src)  python complete_pipeline.py --all  --vi-file D:\NLP\NLP_Trans\data\test.vi --en-file D:\NLP\NLP_Trans\data\test.en  --test-samples 500  --use-comet  --comet-gpus 0

chú ý: có thể thay đường dẫn trên bằng các file test khác tương ứng, --test-samples 500 (số mẫu muốn chọn ra)

Chúng em train trên kaggle - file notebook nằm ở notebooks/

data_preprocessing_v2.py      # Xử lý dữ liệu
tokenizer_sentencepiece.py    # Tokenization
transformer_components.py     # Các thành phần Transformer
transformer_encoder_decoder.py # Encoder & Decoder
complete_transformer.py       # Model hoàn chỉnh
dataloader_module.py          # DataLoader
training_module.py            # Huấn luyện
inference_evaluation_v2.py    # Đánh giá
complete_pipeline.py          # Pipeline tính BLEU và COMMET SCORE
config.py                     