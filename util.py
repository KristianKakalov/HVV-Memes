import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models
from PIL import Image
from transformers import BertModel
import spacy
from config import Config

class FocalLoss(nn.Module):
    def __init__(self, alpha=Config.FOCAL_LOSS_ALPHA, gamma=Config.FOCAL_LOSS_GAMMA, reduction='mean'):
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha.to(inputs.device) * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        return focal_loss

class MemeDataset(Dataset):
    def __init__(self, annotation_path, image_dir, tokenizer, transform):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.samples = []
        self.nlp = spacy.load("en_core_web_sm")

        with open(annotation_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self._process_item(item)

    def _process_item(self, item):
        ocr_text = item["OCR"]
        image_path = self.image_dir / item["image"]
        doc = self.nlp(ocr_text)
        
        entities = list(set(
            ent.text.strip().lower() 
            for ent in doc.ents 
            if ent.text.strip()
        ))

        # default: other
        labels = {ent: [0, 0, 0, 1] for ent in entities}  

        for role in ["hero", "villain", "victim"]:
            for ent in item.get(role, []):
                ent = ent.strip().lower()
                if ent not in labels:
                    labels[ent] = [0, 0, 0, 0]
                labels[ent][Config.LABEL2IDX[role]] = 1
                labels[ent][Config.LABEL2IDX["other"]] = 0

        for ent, label in labels.items():
            self.samples.append((
                ocr_text,
                image_path,
                ent,
                torch.tensor(label, dtype=torch.float)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ocr_text, image_path, entity, label = self.samples[idx]
        
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        text = f"{ocr_text} [ENTITY] {entity}"
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LENGTH,
            return_tensors="pt"
        )
        
        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label
        }

class MultiModalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(Config.BERT_MODEL_NAME)
        
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()
        
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.fc = nn.Sequential(
            nn.Linear(768 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, len(Config.ROLE_LABELS))
        )

    def forward(self, input_ids, attention_mask, image):
        text_feat = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output
        
        img_feat = self.cnn(image)
        combined = torch.cat((text_feat, img_feat), dim=1)
        combined = self.dropout(combined)
        return self.fc(combined)