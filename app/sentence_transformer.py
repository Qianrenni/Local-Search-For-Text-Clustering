from sentence_transformers import SentenceTransformer
from typing import Literal
from config import SETTING
import numpy as np
sentence_transformer_path = SETTING.DEPENDENCY


def get_sentence_transformer(
        model_name:Literal[
            'all-MiniLM-L12-v2',
            'all-MiniLM-L6-v2',
            'all-mpnet-base-v2',
            'clip-ViT-B-32-multilingual-v1',
            'paraphrase-multilingual-MiniLM-L12-v2',
            'qwen'
            ]='all-MiniLM-L6-v2',
        device=None
    ):
    model = SentenceTransformer(str(sentence_transformer_path/model_name),device=device,trust_remote_code=True)
    return model


if __name__ == '__main__':
    # import torch
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # import numpy as np
    # model = get_sentence_transformer('all-MiniLM-L6-v2',device=device)
    # texts= ['实验','experiment','数据','statistics']
    # codes = model.encode(texts)
    # print(model.similarity(codes, codes))
    a = np.array([[1,2,3],[4,5,6]])
    b = np.array([[1,2,3],[4,5,6]])
    print(similarity(a,b))
