import torch
import torch.nn.functional as F
import numpy as np
import argparse
from .model import HAN
from .config import MAX_SENTENCES, MAX_WORDS, MODEL_SAVE_PATH, VOCAB_PATH
from .utils import split_sentences, tokenize, load_vocab

GENRES = ['Music', 'Death', 'Affection', 'Environment']

def poem_to_matrix(poem, vocab):
    sents = split_sentences(poem)[:MAX_SENTENCES]
    matrix = np.zeros((MAX_SENTENCES, MAX_WORDS), dtype=int)
    for i, sent in enumerate(sents):
        tokens = tokenize(sent)[:MAX_WORDS]
        for j, w in enumerate(tokens):
            matrix[i,j] = vocab.get(w, 1)
    return matrix

def predict(poem_text, model, vocab, device):
    matrix = poem_to_matrix(poem_text, vocab)
    tensor = torch.tensor(matrix, dtype=torch.long).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits, w_att, s_att = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
    return GENRES[pred_class], probs, w_att.cpu().numpy(), s_att.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = load_vocab(VOCAB_PATH)
    model = HAN(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    genre, probs, _, _ = predict(args.text, model, vocab, device)
    print(f"Predicted genre: {genre}")
    for i, g in enumerate(GENRES):
        print(f"  {g}: {probs[i]:.4f}")