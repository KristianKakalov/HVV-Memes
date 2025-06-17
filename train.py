# train.py
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from config import Config
from util import MemeDataset, MultiModalClassifier, FocalLoss

def train():
    Config.setup_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize tokenizer and datasets
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
    train_dataset = MemeDataset(Config.TRAIN_ANNOTATIONS, Config.IMAGE_DIR, tokenizer, transform)
    val_dataset = MemeDataset(Config.VAL_ANNOTATIONS, Config.IMAGE_DIR, tokenizer, transform)

    # Handle class imbalance
    label_counts = Counter()
    for _, _, _, label in train_dataset.samples:
        idx = label.argmax().item()
        label_counts[idx] += 1
    
    weights = [
        sum(label_counts.values()) / label_counts[label.argmax().item()] 
        for _, _, _, label in train_dataset.samples
    ]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE
    )

    # Model setup
    model = MultiModalClassifier().to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=2, 
        factor=0.1
    )

    best_val_acc = 0
    for epoch in range(Config.NUM_EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)

            outputs = model(**inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation phase
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
                labels = batch["label"].cpu().numpy()

                logits = model(**inputs)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        # Metrics and logging
        avg_loss = total_loss / len(train_loader)
        val_acc = (sum((pred.argmax() == label.argmax()) 
                  for pred, label in zip(all_preds, all_labels))) / len(all_labels)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            all_labels, all_preds, 
            target_names=Config.ROLE_LABELS, 
            zero_division=0
        ))

        # Confusion matrix
        cm = confusion_matrix(
            [label.argmax() for label in all_labels],
            [pred.argmax() for pred in all_preds]
        )
        plt.figure(figsize=(10, 7))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.colorbar()
        plt.xticks(range(len(Config.ROLE_LABELS)), Config.ROLE_LABELS)
        plt.yticks(range(len(Config.ROLE_LABELS)), Config.ROLE_LABELS)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        for i in range(len(Config.ROLE_LABELS)):
            for j in range(len(Config.ROLE_LABELS)):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        plt.savefig(Config.MATRIX_DIR / f'confusion_epoch_{epoch+1}.png')
        plt.close()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                Config.MODEL_DIR / "meme_classifier_best.pt"
            )
            print("Saved new best model.")

    # Save final model
    torch.save(
        model.state_dict(),
        Config.MODEL_DIR / "meme_classifier_final.pt"
    )

if __name__ == "__main__":
    train()