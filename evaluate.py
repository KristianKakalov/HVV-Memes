# evaluate.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from config import Config
from train import MemeDataset, MultiModalClassifier

class MemeDatasetEval(MemeDataset):
    """Evaluation dataset class (inherits from training dataset)"""
    pass

def evaluate():
    Config.setup_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize tokenizer and dataset
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
    test_dataset = MemeDatasetEval(Config.TEST_ANNOTATIONS, Config.IMAGE_DIR, tokenizer, transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

    # Load model
    model = MultiModalClassifier().to(device)
    model.load_state_dict(torch.load(Config.MODEL_DIR / "meme_classifier_best.pt"))
    model.eval()

    # Evaluation
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].cpu().numpy()

            logits = model(**inputs)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # Metrics
    print("\nTest Classification Report:")
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
    plt.title('Test Set Confusion Matrix')
    plt.colorbar()
    plt.xticks(range(len(Config.ROLE_LABELS)), Config.ROLE_LABELS)
    plt.yticks(range(len(Config.ROLE_LABELS)), Config.ROLE_LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(len(Config.ROLE_LABELS)):
        for j in range(len(Config.ROLE_LABELS)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.savefig(Config.MATRIX_DIR / 'confusion_test.png')
    plt.close()

if __name__ == "__main__":
    evaluate()