import os
import zipfile
from pathlib import Path

import torch
from torchtext.vocab import GloVe
from .config import EMBEDDING_DIM


def _clear_glove_cache(cache_dir=None):
    cache_locations = [Path.home() / '.vector_cache', Path.cwd() / '.vector_cache']
    if cache_dir is not None:
        cache_locations.insert(0, Path(cache_dir))
    for cache_dir in cache_locations:
        zip_path = cache_dir / 'glove.6B.zip'
        if zip_path.exists():
            try:
                zip_path.unlink()
            except OSError:
                pass


def _load_glove():
    cache_dir = Path(os.getenv('TORCHTEXT_CACHE', Path.cwd() / '.vector_cache_new'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        return GloVe(name='6B', dim=EMBEDDING_DIM, cache=str(cache_dir))
    except (zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, zipfile.BadZipFile) or 'not a zip file' in str(exc).lower():
            _clear_glove_cache(cache_dir)
            return GloVe(name='6B', dim=EMBEDDING_DIM, cache=str(cache_dir))
        raise


def build_embedding_matrix(vocab):
    """Tạo ma trận embedding từ vocab dựa trên GloVe."""
    try:
        glove = _load_glove()
    except Exception as exc:
        raise RuntimeError(
            "Không thể tải GloVe embeddings. Hãy đảm bảo kết nối mạng hoặc cài đặt từ nguồn phù hợp." \
            f" Lỗi: {exc}"
        )

    embedding_matrix = torch.randn(len(vocab), EMBEDDING_DIM)
    embedding_matrix[0].zero_()

    for token, idx in vocab.items():
        if token in glove.stoi:
            embedding_matrix[idx] = glove[token]
    return embedding_matrix
