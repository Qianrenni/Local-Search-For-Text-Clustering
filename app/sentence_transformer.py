from sentence_transformers import SentenceTransformer
from typing import Literal
from config import SETTING
sentence_transformer_path = SETTING.DEPENDENCY/'sentence-transformers'

def get_sentence_transformer(
        model_name:Literal[
            'all-MiniLM-L12-v2',
            'all-MiniLM-L6-v2',
            'all-mpnet-base-v2',
            'clip-ViT-B-32-multilingual-v1',
            'paraphrase-multilingual-MiniLM-L12-v2'
            ]='all-MiniLM-L6-v2',
        device=None
    ):
    model = SentenceTransformer(str(sentence_transformer_path/model_name),device=device)
    return model


if __name__ == '__main__':
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import numpy as np
    model = get_sentence_transformer('all-MiniLM-L6-v2',device=device)
    texts= ['我喜欢你','你爱我吗']
    codes = model.encode(texts)
    print(codes.shape)
    print(model.similarity(codes, codes))
