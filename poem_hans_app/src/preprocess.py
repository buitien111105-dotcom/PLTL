import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from .config import MAX_SENTENCES, MAX_WORDS, DATA_PATH, VOCAB_PATH
from .utils import split_sentences, tokenize, save_vocab

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['Poem'])
    df = df[df['Poem'].str.strip() != '']
    
    genres = ['Music', 'Death', 'Affection', 'Environment']
    df = df[df['Genre'].isin(genres)]
    label_map = {g:i for i,g in enumerate(genres)}
    df['label'] = df['Genre'].map(label_map)
    
    # Xây dựng vocab
    all_words = []
    for poem in df['Poem']:
        for sent in split_sentences(poem):
            all_words.extend(tokenize(sent))
    counter = Counter(all_words)
    vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(20000))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    save_vocab(vocab, VOCAB_PATH)
    
    def poem_to_matrix(poem):
        sents = split_sentences(poem)[:MAX_SENTENCES]
        matrix = np.zeros((MAX_SENTENCES, MAX_WORDS), dtype=int)
        for i, sent in enumerate(sents):
            tokens = tokenize(sent)[:MAX_WORDS]
            for j, w in enumerate(tokens):
                matrix[i,j] = vocab.get(w, 1)
        return matrix
    
    X = np.array([poem_to_matrix(p) for p in df['Poem']])
    y = df['label'].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab