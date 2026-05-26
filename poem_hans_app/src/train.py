import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from collections import Counter
from .model import HAN
from .preprocess import load_and_preprocess_data
from .embeddings import build_embedding_matrix
from .config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, NUM_CLASSES

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    (X_train, y_train), (X_val, y_val), _, vocab = load_and_preprocess_data()
    vocab_size = len(vocab)
    
    class_counts = Counter(y_train.tolist())
    weights = [len(y_train) / class_counts[i] for i in range(NUM_CLASSES)]
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Class counts: {dict(class_counts)}")
    print(f"Class weights: {weights}")
    
    X_train_t = torch.tensor(X_train, dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # Oversample all classes to reduce dataset bias and help underrepresented labels
    # Additionally boost sampling for `Death` (label 1) to counter persistent low recall
    max_count = max(class_counts.values())
    base_weights = [max_count / class_counts[int(lbl)] for lbl in y_train.tolist()]
    # apply extra multiplier for Death (label==1)
    sample_weights = [w * (2.0 if int(lbl) == 1 else 1.0) for w, lbl in zip(base_weights, y_train.tolist())]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE)
    
    print("Building pretrained embedding matrix...")
    embedding_weights = build_embedding_matrix(vocab)
    model = HAN(vocab_size, embedding_weights=embedding_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits, _, _ = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == by).sum().item()
                total += by.size(0)
        val_acc = correct / total
        scheduler.step(val_loss)
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Best model saved (acc={val_acc:.4f})")
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train()