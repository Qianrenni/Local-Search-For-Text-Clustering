from sentence_transformers import SentenceTransformer
from typing import Literal
sentence_transformer_path = r'D:\graduate_design\code\dependency\sentence-transformers\all-MiniLM-L6-v2'
qwen3_embedding_06B_path = r'D:\graduate_design\code\dependency\Qwen3-Embedding-06B'

def get_sentence_transformer(
        path:Literal['qwen3_06B','all_minilm_l6']='all_minilm_l6',
        device=None
    ):
    p = {
        'qwen3_06B':qwen3_embedding_06B_path,
        'all_minilm_l6':sentence_transformer_path
    }
    model = SentenceTransformer(p.get(path),device=device)
    return model

if __name__ == '__main__':
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_sentence_transformer(path='qwen3_06B',device=device)
    texts= ['heloo']
    codes = model.encode(texts)
    print(codes.shape)
