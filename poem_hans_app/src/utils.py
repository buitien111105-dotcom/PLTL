# src/utils.py
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize

# Tự động tải punkt nếu chưa có
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    """Làm sạch văn bản: bỏ dấu ngoặc kép, khoảng trắng thừa"""
    text = re.sub(r'[“”"\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_sentences(poem):
    """Chia bài thơ thành danh sách các câu (mỗi dòng là một câu)"""
    lines = poem.split('\n')
    sentences = []
    for line in lines:
        line = clean_text(line)
        if len(line) > 0:
            sentences.append(line)
    return sentences

def tokenize(sentence):
    """Tokenize một câu thành danh sách từ, viết thường"""
    return word_tokenize(sentence.lower())

def save_vocab(vocab, path):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(path):
    with open(path, 'rb') as f:
        return pickle.load(f)