import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    IMAGE_DIR = DATA_DIR / "images"
    ANNOTATION_DIR = DATA_DIR / "annotations"
    MODEL_DIR = BASE_DIR / "models"
    MATRIX_DIR = BASE_DIR / "matrix"
    
    TRAIN_ANNOTATIONS = ANNOTATION_DIR / "train.jsonl"
    VAL_ANNOTATIONS = ANNOTATION_DIR / "intermediate.jsonl"
    TEST_ANNOTATIONS = ANNOTATION_DIR / "test.jsonl"
    
    BERT_MODEL_NAME = "bert-base-uncased"
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    NUM_EPOCHS = 15
    LEARNING_RATE = 2e-5
    DROPOUT_RATE = 0.3
    
    ROLE_LABELS = ["hero", "villain", "victim", "other"]
    LABEL2IDX = {label: idx for idx, label in enumerate(ROLE_LABELS)}
    
    FOCAL_LOSS_ALPHA = [3.0, 1.5, 2.2, 0.5]
    FOCAL_LOSS_GAMMA = 2
    
    @classmethod
    def setup_directories(cls):
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.MATRIX_DIR, exist_ok=True)
        os.makedirs(cls.IMAGE_DIR, exist_ok=True)
        os.makedirs(cls.ANNOTATION_DIR, exist_ok=True)