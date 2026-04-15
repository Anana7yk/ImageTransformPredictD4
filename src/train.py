import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


# ======================
# Metrics
# ======================

def compute_sequence_accuracy(pred_logits, target_ids, pad_token_id):
    """
    Exact match accuracy: fraction of sequences where all non-padding tokens are predicted correctly.
    """
    pred_ids = pred_logits.argmax(dim=-1)  # [N, T]
    mask = (target_ids != pad_token_id)
    correct = (pred_ids == target_ids) | (~mask)
    seq_correct = correct.all(dim=1)
    return seq_correct.float().mean().item()


def compute_token_accuracy(pred_logits, target_ids, pad_token_id):
    """
    Token-level accuracy: fraction of correctly predicted non-padding tokens.
    """
    pred_ids = pred_logits.argmax(dim=-1)
    mask = (target_ids != pad_token_id)
    if mask.sum() == 0:
        return 1.0
    correct = (pred_ids == target_ids) & mask
    return (correct.sum().float() / mask.sum().float()).item()


# ======================
# Configuration and Checkpoint Management
# ======================

def get_optimizer(net, config):
    """Initialize optimizer based on config."""
    optimizer_name = config['optimizer']['name']
    if optimizer_name == 'Adam':
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=config['optimizer']['lr'],
            betas=config['optimizer']['betas'],
            weight_decay=config['optimizer']['weight_decay']
        )
    elif optimizer_name == 'AdamW':
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=config['optimizer']['lr'],
            betas=config['optimizer']['betas'],
            weight_decay=config['optimizer']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return opt


def get_scheduler(opt, config):
    """Initialize learning rate scheduler."""
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=config['scheduler']['milestones'],
        gamma=config['scheduler']['gamma']
    )
    return sched


def save_checkpoint(model, optimizer, scheduler, epoch, config):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }

    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved at epoch {epoch}')


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint if exists."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Checkpoint loaded from epoch {epoch}')
        return epoch
    else:
        print('No checkpoint found. Starting from scratch.')
        return 0


# ======================
# Main Training Loop
# ======================

def train_model(model, train_loader, val_loader, config):
    """
    Train the autoregressive model on explicit image pairs and target sequences.
    """
    optimizer = get_optimizer(model, config)
    lr_scheduler = get_scheduler(optimizer, config)

    # === Training setup ===
    num_epochs = config['training']['num_epochs']
    device = torch.device(config['training']['device'])
    checkpoint_interval = config['training']['checkpoint_interval']
    checkpoint_dir = config['training']['checkpoint_dir']
    log_dir = config['data']['tensorboard_logdir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    model.to(device)

    pad_token_id = config.model.decoder.pad_token_id

    # === Resume ===
    start_epoch = 0
    if config['training']['resume']:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            latest_checkpoint = max(
                [os.path.join(checkpoint_dir, f) for f in checkpoints],
                key=os.path.getctime
            )
            start_epoch = load_checkpoint(model, optimizer, lr_scheduler, latest_checkpoint)

    # === Training loop ===
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_seq_acc = 0.0
        train_token_acc = 0.0
        total_samples = 0

        for orig_batch, paired_batch, idx_batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            orig_batch = orig_batch.to(device)
            paired_batch = paired_batch.to(device)
            idx_batch = idx_batch.to(device)

            optimizer.zero_grad()
            targets = torch.full_like(idx_batch, pad_token_id)
            targets[:, :-1] = idx_batch[:, 1:]

            logits, loss = model(orig_batch, paired_batch, idx_batch)

            loss.backward()
            optimizer.step()

            batch_seq_acc = compute_sequence_accuracy(logits, targets, pad_token_id)
            batch_token_acc = compute_token_accuracy(logits, targets, pad_token_id)

            batch_size = orig_batch.size(0)
            train_loss += loss.item() * batch_size
            train_seq_acc += batch_seq_acc * batch_size
            train_token_acc += batch_token_acc * batch_size
            total_samples += batch_size

        lr_scheduler.step()

        avg_train_loss = train_loss / total_samples
        avg_train_seq_acc = train_seq_acc / total_samples
        avg_train_token_acc = train_token_acc / total_samples

        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_seq_acc = 0.0
        val_token_acc = 0.0
        val_total = 0

        with torch.no_grad():
            for orig_batch, paired_batch, idx_batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                orig_batch = orig_batch.to(device)
                paired_batch = paired_batch.to(device)
                idx_batch = idx_batch.to(device)

                targets = torch.full_like(idx_batch, pad_token_id)
                targets[:, :-1] = idx_batch[:, 1:]

                logits, loss = model(orig_batch, paired_batch, idx_batch)

                batch_seq_acc = compute_sequence_accuracy(logits, targets, pad_token_id)
                batch_token_acc = compute_token_accuracy(logits, targets, pad_token_id)

                batch_size = orig_batch.size(0)
                val_loss += loss.item() * batch_size
                val_seq_acc += batch_seq_acc * batch_size
                val_token_acc += batch_token_acc * batch_size
                val_total += batch_size

        avg_val_loss = val_loss / val_total
        avg_val_seq_acc = val_seq_acc / val_total
        avg_val_token_acc = val_token_acc / val_total

        # === Logging ===
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f} | SeqAcc: {avg_train_seq_acc:.4f} | TokAcc: {avg_train_token_acc:.4f}')
        print(f'  Val   Loss: {avg_val_loss:.4f} | SeqAcc: {avg_val_seq_acc:.4f} | TokAcc: {avg_val_token_acc:.4f}')
        print(f'  LR: {current_lr:.2e}\n')

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('SeqAcc/Train', avg_train_seq_acc, epoch)
        writer.add_scalar('TokAcc/Train', avg_train_token_acc, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('SeqAcc/Val', avg_val_seq_acc, epoch)
        writer.add_scalar('TokAcc/Val', avg_val_token_acc, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)

        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, lr_scheduler, epoch + 1, config)

    writer.close()
    print("Training finished.")
