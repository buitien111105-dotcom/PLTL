import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Poem_classification - train_data.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "best_han_model.pt")
VOCAB_PATH = os.path.join(BASE_DIR, "models", "vocab.pkl")

MAX_SENTENCES = 30
MAX_WORDS = 30
EMBEDDING_DIM = 128
HIDDEN_SIZE = 64
NUM_CLASSES = 4
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001