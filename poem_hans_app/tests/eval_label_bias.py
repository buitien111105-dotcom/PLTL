import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from src.predict import predict, GENRES
from src.utils import load_vocab
from src.model import HAN
from src.config import MODEL_SAVE_PATH, VOCAB_PATH
import torch


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = load_vocab(VOCAB_PATH)
    model = HAN(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    return model, vocab, device


def eval_on_label(label, n=100):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'Poem_classification - train_data.csv'))
    samples = df[df['Genre']==label]['Poem'].dropna().iloc[:n]
    model, vocab, device = load_model()
    counts = {g:0 for g in GENRES}
    for text in samples:
        genre, probs, _, _ = predict(text, model, vocab, device)
        counts[genre]+=1
    return counts

if __name__=='__main__':
    for lab in ['Death','Affection']:
        counts = eval_on_label(lab, n=50)
        print('Label:', lab, '->', counts)
