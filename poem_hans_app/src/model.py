import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import EMBEDDING_DIM, HIDDEN_SIZE, NUM_CLASSES, DROPOUT

class WordAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Linear(hidden_size, 1, bias=False)
    def forward(self, lstm_out):
        u = torch.tanh(self.weight(lstm_out))
        att = self.context(u).squeeze(-1)
        att_weights = F.softmax(att, dim=1)
        weighted = lstm_out * att_weights.unsqueeze(-1)
        return weighted.sum(dim=1), att_weights

class SentenceAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Linear(hidden_size, 1, bias=False)
    def forward(self, lstm_out):
        u = torch.tanh(self.weight(lstm_out))
        att = self.context(u).squeeze(-1)
        att_weights = F.softmax(att, dim=1)
        weighted = lstm_out * att_weights.unsqueeze(-1)
        return weighted.sum(dim=1), att_weights

class HAN(nn.Module):
    def __init__(self, vocab_size, embedding_weights=None):
        super().__init__()
        if embedding_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weights, padding_idx=0, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.dropout = nn.Dropout(DROPOUT)
        self.word_lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, bidirectional=True, batch_first=True)
        self.word_att = WordAttention(HIDDEN_SIZE*2)
        self.sent_lstm = nn.LSTM(HIDDEN_SIZE*2, HIDDEN_SIZE, bidirectional=True, batch_first=True)
        self.sent_att = SentenceAttention(HIDDEN_SIZE*2)
        self.fc = nn.Linear(HIDDEN_SIZE*2, NUM_CLASSES)

    def forward(self, x):
        batch_size, max_sents, max_words = x.shape
        x = x.view(batch_size*max_sents, max_words)
        emb = self.dropout(self.embedding(x))
        word_out, _ = self.word_lstm(emb)
        sent_vecs, w_att = self.word_att(word_out)
        sent_vecs = sent_vecs.view(batch_size, max_sents, -1)
        sent_out, _ = self.sent_lstm(sent_vecs)
        doc_vec, s_att = self.sent_att(sent_out)
        logits = self.fc(doc_vec)
        return logits, w_att, s_att