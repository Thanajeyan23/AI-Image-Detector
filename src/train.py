import torch
import os
import sys

# Allow running from project root OR from src/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model import get_model


# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH  = os.path.join(MODELS_DIR, 'detector.pth')

BATCH_SIZE  = 64     # Works well for RTX 3050 6GB
EPOCHS      = 10     # 10 passes through the full dataset
LR          = 0.001  # Learning rate — how fast the model updates
PATIENCE    = 3      # Stop early if val accuracy doesn't improve for 3 epochs

os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Device Setup ─────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{'='*50}")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"{'='*50}\n")

# ─── Data ─────────────────────────────────────────────────────────────────────
train_loader, val_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

# ─── Model, Loss, Optimizer ───────────────────────────────────────────────────
model     = get_model().to(device)
criterion = torch.nn.CrossEntropyLoss()

# Only optimize parameters that are NOT frozen (i.e., layer4 + fc)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)

# Reduce learning rate when val accuracy stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, verbose=True
)

# Mixed precision — uses less VRAM and trains faster on RTX cards
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))


# ─── Training Functions ────────────────────────────────────────────────────────
def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        predicted  = outputs.argmax(dim=1)
        correct   += (predicted == labels).sum().item()
        total     += labels.size(0)

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {total_loss/(batch_idx+1):.4f} "
                  f"| Acc: {100*correct/total:.1f}%")

    return total_loss / len(train_loader), 100 * correct / total


def validate():
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            predicted   = outputs.argmax(dim=1)
            correct    += (predicted == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(val_loader), 100 * correct / total


# ─── Training Loop ─────────────────────────────────────────────────────────────
print("Starting training...\n")
best_val_acc   = 0.0
patience_count = 0

for epoch in range(EPOCHS):
    print(f"\n" + "-"*50)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("-"*50)

    train_loss, train_acc = train_one_epoch(epoch)
    val_loss,   val_acc   = validate()

    scheduler.step(val_acc)

    print(f"  [Train] Loss: {train_loss:.4f} | Acc: {train_acc:.1f}%")
    print(f"  [Val]   Loss: {val_loss:.4f} | Acc: {val_acc:.1f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  >> New best model saved! (Val Acc: {val_acc:.1f}%)")
        patience_count = 0
    else:
        patience_count += 1
        print(f"  -- No improvement ({patience_count}/{PATIENCE})")
        if patience_count >= PATIENCE:
            print("\nEarly stopping triggered - training complete!")
            break

print("\n" + "="*50)
print("Training finished!")
print(f"Best Validation Accuracy: {best_val_acc:.1f}%")
print(f"Model saved to: {MODEL_PATH}")
print("="*50)