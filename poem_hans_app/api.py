from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from src.model import HAN
from src.predict import predict, GENRES
from src.config import MODEL_SAVE_PATH, VOCAB_PATH
from src.utils import load_vocab

app = FastAPI(title="Poem Classifier API")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = load_vocab(VOCAB_PATH)
model = HAN(vocab_size=len(vocab)).to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

class PoemRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_endpoint(req: PoemRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty poem text")
    genre, probs, _, _ = predict(req.text, model, vocab, device)
    return {"genre": genre, "probabilities": {g: float(probs[i]) for i, g in enumerate(GENRES)}}

@app.get("/")
def root():
    return {"message": "Poem Classifier API using HAN"}