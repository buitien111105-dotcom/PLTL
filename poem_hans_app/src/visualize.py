# src/visualize.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from .model import HAN
from .config import MAX_SENTENCES, MAX_WORDS, MODEL_SAVE_PATH, VOCAB_PATH
from .utils import load_vocab, split_sentences, tokenize
from .predict import poem_to_matrix, GENRES

def visualize_attention(poem_text, model_path=MODEL_SAVE_PATH, save_fig=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vocab = load_vocab(VOCAB_PATH)
    model = HAN(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    matrix = poem_to_matrix(poem_text, vocab)
    tensor = torch.tensor(matrix, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, word_att, sent_att = model(tensor)
    
    # Xử lý shape: 
    # word_att: (batch * max_sents, max_words) -> với batch=1 => (max_sents, max_words)
    # sent_att: (batch, max_sents) -> (max_sents,)
    word_att = word_att.cpu().numpy()
    sent_att = sent_att.cpu().numpy()
    
    if sent_att.ndim == 2:
        sent_att = sent_att[0]          # lấy batch đầu tiên
    if word_att.ndim == 3:
        word_att = word_att[0]          # nếu có batch dimension dư
    
    # Lấy danh sách câu thực tế
    sentences = split_sentences(poem_text)[:MAX_SENTENCES]
    num_sents = len(sentences)
    
    # Đảm bảo sent_att có độ dài ít nhất num_sents
    if len(sent_att) < num_sents:
        # Pad bằng 0 nếu thiếu
        sent_att = np.pad(sent_att, (0, num_sents - len(sent_att)), 'constant')
    
    # Vẽ Sentence Attention
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(num_sents), sent_att[:num_sents], color='skyblue', edgecolor='navy')
    plt.xlabel('Câu thứ tự')
    plt.ylabel('Trọng số Attention')
    plt.title('Sentence-level Attention')
    plt.xticks(range(num_sents), [f'S{i+1}' for i in range(num_sents)], rotation=45)
    
    # Tìm câu có trọng số cao nhất
    best_sent_idx = np.argmax(sent_att[:num_sents])
    best_sent = sentences[best_sent_idx]
    tokens = tokenize(best_sent)[:MAX_WORDS]
    
    # Lấy word attention cho câu đó
    if word_att.shape[0] > best_sent_idx:
        att_weights = word_att[best_sent_idx][:len(tokens)]
    else:
        att_weights = np.zeros(len(tokens))
    
    # Vẽ Word Attention
    plt.subplot(1, 2, 2)
    colors = plt.cm.viridis(att_weights / (att_weights.max() + 1e-8))
    plt.barh(range(len(tokens)), att_weights, color=colors, edgecolor='gray')
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel('Trọng số')
    plt.title(f'Word Attention (câu S{best_sent_idx+1})')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    if save_fig:
        plt.savefig('attention_visualization.png', dpi=150)
        print("Đã lưu hình ảnh: attention_visualization.png")
    else:
        plt.show()
    
    # In thông tin text
    print("\n=== Sentence Attention Scores ===")
    for i, (sent, score) in enumerate(zip(sentences, sent_att[:num_sents])):
        print(f"S{i+1}: {score:.4f} | {sent[:80]}{'...' if len(sent)>80 else ''}")
    
    print(f"\n=== Câu quan trọng nhất (S{best_sent_idx+1}) ===")
    print(best_sent)
    print("\n=== Từ quan trọng nhất trong câu đó ===")
    top_words = sorted(zip(tokens, att_weights), key=lambda x: x[1], reverse=True)[:5]
    for word, score in top_words:
        print(f"  {word}: {score:.4f}")

def visualize_from_file(poem_file, model_path=MODEL_SAVE_PATH):
    with open(poem_file, 'r', encoding='utf-8') as f:
        poem_text = f.read()
    visualize_attention(poem_text, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--poem', type=str, help='Poem text')
    parser.add_argument('--file', type=str, help='Path to text file')
    parser.add_argument('--model', type=str, default=MODEL_SAVE_PATH)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    
    if args.file:
        visualize_from_file(args.file, args.model)
    elif args.poem:
        visualize_attention(args.poem, args.model, args.save)
    else:
        print("Vui lòng cung cấp --poem hoặc --file")
        print("Ví dụ: python -m src.visualize --poem \"The sun sets over the hill\\nAnd the birds are still\"")