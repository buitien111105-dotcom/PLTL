import sys, os
# ensure poem_hans_app is on sys.path when running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict, GENRES
from src.utils import load_vocab
from src.config import MODEL_SAVE_PATH, VOCAB_PATH
from src.model import HAN
import torch

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = load_vocab(VOCAB_PATH)
    model = HAN(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    return model, vocab, device

examples = {
    'death_1': "          in memory of Seth Walsh, Justin Aaberg, Billy Lucas, and Tyler ClementiThere are those who suffer in plain sight, there are those who suffer in private. Nothing but secondhand details: a last shower, a request for a pen, a tall red oak.There are those who suffer in private. The one in Tehachapi, aged 13. A",
    'death_2': " Lo! Death has reared himself a throneIn a strange city lying aloneFar down within the dim West,Wherethe good and the bad and the worst and the best",
    'aff_1': "Through the dark pine trunks Silver and yellow gleam the clouds And the sun; The sea is faint purple. My love, my love, I shall never reach you.",
    'aff_2': "for TomicaMy love is as ancient as my blood. And of course my blood is still mine because a woman, sweetened black with good song, pulled me from the river like an axe pulled back from the bark. I learned love, first, as scar. And of course my love is"
}

if __name__ == '__main__':
    model, vocab, device = load_model()
    for name, text in examples.items():
        genre, probs, w_att, s_att = predict(text, model, vocab, device)
        # get logits as well by re-running model forward
        import torch
        tensor = torch.tensor(__import__('numpy').zeros((1,30,30),dtype=int))
        # create correct tensor using poem_to_matrix from src.predict
        from src.predict import poem_to_matrix
        import numpy as np
        matrix = poem_to_matrix(text, vocab)
        tensor = torch.tensor(matrix, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, w_att2, s_att2 = model(tensor)
        logits_np = logits.cpu().numpy()[0]

        print(f"== {name} ==")
        print("Logits:", [float(x) for x in logits_np])
        print("Predicted:", genre)
        for i, g in enumerate(GENRES):
            print(f"  {g}: {probs[i]:.4f}")
        print("Sentence attention (first 10):", s_att2.cpu().numpy()[0][:10])
        print()
