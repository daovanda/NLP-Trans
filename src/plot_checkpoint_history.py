import os
import json

# ==== FIX LỖI OMP DUPLICATE LIB ====
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ==================================

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_history(ckpt_path):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    history = ckpt.get("history", None)
    if history is None:
        raise ValueError(f"'history' not found in checkpoint: {ckpt_path}")
    return history


def export_history_json(history, save_path):
    """Ghi toàn bộ lịch sử + phân tích nhỏ ra JSON."""
    summary = {
        "num_epochs": len(history["train_loss"]),
        "best_val_loss": min(history["val_loss"]),
        "best_epoch": history["val_loss"].index(min(history["val_loss"])) + 1,
        "metrics": {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_ppl": history["train_ppl"],
            "val_ppl": history["val_ppl"],
            "learning_rate": history["lr"],
        }
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved JSON: {save_path}")


def plot_all_in_one(history, save_path, title=""):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 10))

    # === SUBPLOT 1: Train vs Val Loss ===
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker='o')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # === SUBPLOT 2: Train vs Val PPL ===
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["train_ppl"], label="Train PPL", marker='x')
    plt.plot(epochs, history["val_ppl"], label="Val PPL", marker='x')
    plt.title("Perplexity (PPL)")
    plt.xlabel("Epoch")
    plt.ylabel("PPL")
    plt.grid(True)
    plt.legend()

    # === SUBPLOT 3: Learning Rate ===
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["lr"], label="Learning Rate", linestyle="--")
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.grid(True)
    plt.legend()

    # === SUBPLOT 4: Summary Overlay ===
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.plot(epochs, history["train_ppl"], label="Train PPL")
    plt.plot(epochs, history["val_ppl"], label="Val PPL")
    plt.title("Overlay Summary")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    ckpt_en_vi = "checkpoints/en_vi/best_en_vi_model.pt"
    ckpt_vi_en = "checkpoints/vi_en/best_vi_en_model.pt"

    output_dir = "training_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Load histories
    hist_en_vi = load_history(ckpt_en_vi)
    hist_vi_en = load_history(ckpt_vi_en)

    # === Vẽ hình ===
    plot_all_in_one(
        hist_en_vi,
        save_path=f"{output_dir}/EN_VI_all_metrics.png",
        title="EN → VI: Train Loss, Val Loss, PPL, LR"
    )

    plot_all_in_one(
        hist_vi_en,
        save_path=f"{output_dir}/VI_EN_all_metrics.png",
        title="VI → EN: Train Loss, Val Loss, PPL, LR"
    )

    # === Export JSON ===
    export_history_json(hist_en_vi, f"{output_dir}/EN_VI_history.json")
    export_history_json(hist_vi_en, f"{output_dir}/VI_EN_history.json")

    print("\n🎉 Done! Plots + JSON saved in /training_plots")
