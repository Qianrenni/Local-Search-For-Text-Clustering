from sentence_transformers import SentenceTransformer
from typing import Literal
sentence_transformer_path = r'D:\graduate_design\code\dependency\sentence-transformers\all-MiniLM-L6-v2'
gte_path = r'D:\graduate_design\code\dependency\gte_sentence-embedding_multilingual-base'

def get_sentence_transformer(
        path:Literal['gte','all_minilm_l6']='all_minilm_l6',
        device=None
    ):
    p = {
        'gte':gte_path,
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
