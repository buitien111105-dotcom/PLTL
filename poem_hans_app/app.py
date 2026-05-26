import streamlit as st
import torch
from src.model import HAN
from src.predict import predict, GENRES
from src.config import MODEL_SAVE_PATH, VOCAB_PATH
from src.utils import load_vocab
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Poem Classifier with HAN", layout="wide")
st.title("📜 Phân loại thể loại thơ bằng Hierarchical Attention Network")

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = load_vocab(VOCAB_PATH)
    model = HAN(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    return model, vocab, device

model, vocab, device = load_model()

poem_input = st.text_area("Nhập bài thơ (mỗi dòng là một câu):", height=300)

if st.button("Phân loại"):
    if poem_input.strip():
        genre, probs, w_att, s_att = predict(poem_input, model, vocab, device)
        st.success(f"**Thể loại dự đoán:** {genre}")
        st.subheader("Xác suất các thể loại:")
        for i, g in enumerate(GENRES):
            st.write(f"- {g}: {probs[i]:.4f}")
        
        # Hiển thị sentence attention
        st.subheader("Attention cấp câu")
        sentences = poem_input.split('\n')[:30]
        sent_att = s_att[0][:len(sentences)]
        fig, ax = plt.subplots(figsize=(10,4))
        ax.bar(range(len(sentences)), sent_att)
        ax.set_xticks(range(len(sentences)))
        ax.set_xticklabels([f"S{i+1}" for i in range(len(sentences))], rotation=45)
        ax.set_ylabel("Trọng số")
        st.pyplot(fig)
    else:
        st.warning("Vui lòng nhập nội dung bài thơ.")