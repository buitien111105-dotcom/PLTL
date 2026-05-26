import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.preprocess import load_and_preprocess_data
from src.model import HAN
from src.utils import load_vocab
from src.config import MODEL_SAVE_PATH
import torch


def evaluate():
    (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab = load_and_preprocess_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HAN(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.long).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    with torch.no_grad():
        logits, _, _ = model(X_test_t)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, preds)
    print('Confusion matrix:')
    print(cm)
    print('\nClassification report:')
    print(classification_report(y_test, preds, target_names=['Music','Death','Affection','Environment']))

if __name__=='__main__':
    evaluate()
