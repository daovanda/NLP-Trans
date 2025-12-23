import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import autocast, GradScaler
import math
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. LABEL SMOOTHING CROSS ENTROPY LOSS

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Cross Entropy Loss"""
    def __init__(self, vocab_size, pad_idx=0, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        batch_size, seq_len, vocab_size = pred.size()
        pred = pred.reshape(-1, vocab_size)
        target = target.reshape(-1)
        
        log_probs = torch.log_softmax(pred, dim=-1)
        smoothed_targets = torch.zeros_like(log_probs)
        smoothed_targets.fill_(self.smoothing / (vocab_size - 2))
        smoothed_targets.scatter_(1, target.unsqueeze(1), self.confidence)
        smoothed_targets[:, self.pad_idx] = 0
        
        mask = (target != self.pad_idx).float()
        loss = -(smoothed_targets * log_probs).sum(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss

# 2. LEARNING RATE SCHEDULER

class TransformerLRScheduler(_LRScheduler):
    """Learning Rate Scheduler với Warmup"""
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.num_steps = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        self.num_steps += 1
        lr = self.factor * (
            self.d_model ** (-0.5) *
            min(self.num_steps ** (-0.5), 
                self.num_steps * self.warmup_steps ** (-1.5))
        )
        return [lr for _ in self.base_lrs]

# 3. PERPLEXITY METRIC

def calculate_perplexity(loss):
    """Tính Perplexity từ loss"""
    return math.exp(min(loss, 100))

# 4. SAVE/LOAD CHECKPOINT - OPTIMIZED

def save_checkpoint(model, optimizer, scheduler, epoch, batch_idx, 
                   train_loss, val_loss, checkpoint_dir, history=None, 
                   scaler=None, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'history': history
    }
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # Lưu checkpoint theo epoch và batch
    checkpoint_path = os.path.join(
        checkpoint_dir, 
        f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pt'
    )
    torch.save(checkpoint, checkpoint_path)
    
    # Lưu latest checkpoint (để resume)
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    torch.save(checkpoint, latest_path)
    
    # Nếu là best model, lưu riêng
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        return checkpoint_path, best_path
    
    return checkpoint_path, None

def load_checkpoint(model, checkpoint_path, device, optimizer=None, 
                   scheduler=None, scaler=None):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    batch_idx = checkpoint.get('batch_idx', 0)
    history = checkpoint.get('history', {
        'train_loss': [], 'train_ppl': [],
        'val_loss': [], 'val_ppl': [], 'lr': []
    })
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    print(f"✓ Resumed from Epoch {epoch}, Batch {batch_idx}")
    print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    
    return model, optimizer, scheduler, epoch, batch_idx, history, scaler

# 5. TRAINING FUNCTION - OPTIMIZED WITH MIXED PRECISION

def train_epoch(model, train_loader, optimizer, scheduler, criterion, 
                device, epoch, checkpoint_dir, save_every_batches=500,
                use_amp=True, history=None, start_batch=0):
    model.train()
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None
    
    total_loss = 0
    total_tokens = 0
    batch_loss = 0
    batch_tokens = 0
    
    progress_bar = tqdm(
        enumerate(train_loader, start=start_batch), 
        desc=f'Epoch {epoch}',
        total=len(train_loader)
    )
    
    for batch_idx, (src, tgt, src_len, tgt_len) in progress_bar:
        # Move to device
        src = src.to(device)
        tgt = tgt.to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        optimizer.zero_grad()
        
        # Forward pass với mixed precision
        if use_amp:
            with autocast():
                output = model(src, tgt_input)
                loss = criterion(output, tgt_output)
            
            # Backward với gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(src, tgt_input)
            loss = criterion(output, tgt_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Statistics
        num_tokens = (tgt_output != criterion.pad_idx).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        batch_loss += loss.item() * num_tokens
        batch_tokens += num_tokens
        
        # Update progress bar
        current_loss = loss.item()
        current_ppl = calculate_perplexity(current_loss)
        current_lr = scheduler.get_last_lr()[0]
        
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'ppl': f'{current_ppl:.2f}',
            'lr': f'{current_lr:.6f}'
        })
        
        # Lưu checkpoint theo batch
        if (batch_idx + 1) % save_every_batches == 0:
            avg_batch_loss = batch_loss / batch_tokens
            print(f"\n Saving checkpoint at batch {batch_idx + 1}...")
            
            checkpoint_path, _ = save_checkpoint(
                model, optimizer, scheduler, epoch, batch_idx + 1,
                avg_batch_loss, None, checkpoint_dir, history, scaler
            )
            print(f"✓ Saved: {checkpoint_path}")
            
            # Reset batch statistics
            batch_loss = 0
            batch_tokens = 0
    
    avg_loss = total_loss / total_tokens
    avg_perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, avg_perplexity, scaler

# 6. VALIDATION FUNCTION

def validate(model, val_loader, criterion, device):
    """Đánh giá trên validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for src, tgt, src_len, tgt_len in tqdm(val_loader, desc='Validation'):
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = model(src, tgt_input)
            loss = criterion(output, tgt_output)
            
            num_tokens = (tgt_output != criterion.pad_idx).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    avg_perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, avg_perplexity

# 7. MAIN TRAINING LOOP - OPTIMIZED

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    d_model,
    warmup_steps=4000,
    label_smoothing=0.1,
    checkpoint_dir='checkpoints',
    save_every=1,
    save_every_batches=500,
    start_epoch=1,
    use_amp=True
):
   
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check xem có checkpoint để resume không
    latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    start_batch = 0
    scaler = None
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # Setup scheduler
    scheduler = TransformerLRScheduler(
        optimizer,
        d_model=d_model,
        warmup_steps=warmup_steps
    )
    
    # Setup loss
    criterion = LabelSmoothingLoss(
        vocab_size=model.decoder.fc_out.out_features,
        pad_idx=model.pad_idx,
        smoothing=label_smoothing
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_ppl': [],
        'val_loss': [],
        'val_ppl': [],
        'lr': []
    }
    
    # Thử load checkpoint nếu có
    if os.path.exists(latest_checkpoint):
        print("Found existing checkpoint. Resuming training...")
        model, optimizer, scheduler, start_epoch, start_batch, history, scaler = \
            load_checkpoint(model, latest_checkpoint, device, optimizer, 
                          scheduler, GradScaler() if use_amp else None)
        print(f"  Resuming from Epoch {start_epoch}, Batch {start_batch}")
    
    best_val_loss = float('inf')
    
    print("="*70)
    print("BẮT ĐẦU HUẤN LUYỆN")
    print("="*70)
    print(f"Device: {device}")
    print(f"Mixed Precision (AMP): {use_amp}")
    print(f"Save checkpoint every: {save_every_batches} batches")
    print(f"Total batches per epoch: {len(train_loader)}")
    print(f"Estimated checkpoint saves per epoch: {len(train_loader) // save_every_batches}")
    print("="*70 + "\n")
    
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        
        # Training
        train_loss, train_ppl, scaler = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, 
            device, epoch, checkpoint_dir, save_every_batches,
            use_amp, history, start_batch if epoch == start_epoch else 0
        )
        
        # Reset start_batch sau epoch đầu
        if epoch == start_epoch:
            start_batch = 0
        
        # Validation
        val_loss, val_ppl = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start_time
        
        # Lưu history
        history['train_loss'].append(train_loss)
        history['train_ppl'].append(train_ppl)
        history['val_loss'].append(val_loss)
        history['val_ppl'].append(val_ppl)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        # In kết quả
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{num_epochs} - Time: {epoch_time/60:.2f} minutes")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Lưu checkpoint cuối epoch
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  New best model!")
        
        checkpoint_path, best_path = save_checkpoint(
            model, optimizer, scheduler, epoch, 0,
            train_loss, val_loss, checkpoint_dir, history, scaler, is_best
        )
        print(f"  ✓ Saved checkpoint: {checkpoint_path}")
        if best_path:
            print(f"  ✓ Saved best model: {best_path}")
        print("="*70)
    
    print("\n" + "="*70)
    print("🎉 HOÀN THÀNH HUẤN LUYỆN")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {calculate_perplexity(best_val_loss):.2f}")
    
    return history

# 8. PLOT TRAINING HISTORY


def plot_training_history(history, save_path='training_history.png'):
    """Vẽ đồ thị training history"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Perplexity
    axes[1].plot(epochs, history['train_ppl'], 'b-', label='Train PPL')
    axes[1].plot(epochs, history['val_ppl'], 'r-', label='Val PPL')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Training and Validation Perplexity')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot Learning Rate
    axes[2].plot(epochs, history['lr'], 'g-')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved training history plot to {save_path}")
    plt.close()