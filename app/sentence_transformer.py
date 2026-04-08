from sentence_transformers import SentenceTransformer
from typing import Literal
from config import SETTING
import os
import dashscope
import numpy as np
sentence_transformer_path = SETTING.DEPENDENCY/'sentence-transformers'

def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    计算两组向量之间的余弦相似度矩阵。

    Args:
        a (np.ndarray): 形状为 (n_samples_a, n_features) 的向量数组
        b (np.ndarray): 形状为 (n_samples_b, n_features) 的向量数组

    Returns:
        np.ndarray: 形状为 (n_samples_a, n_samples_b) 的余弦相似度矩阵
    """
    # 计算 L2 范数，增加 epsilon 避免除以零
    normalized_a = a /  np.sqrt(np.sum(a**2, axis=1, keepdims=True))
    normalized_b = b / np.sqrt(np.sum(b**2, axis=1, keepdims=True))
    
    # 计算余弦相似度矩阵
    return normalized_a @ normalized_b.T
class QwenEmbedding:
    def __init__(self):
        pass
    def encode(self,texts:list[str],normalize_embeddings:bool=False):
        resp = dashscope.TextEmbedding.call(
                model="text-embedding-v4",
                input=texts,
                api_key=os.getenv('DASHSCOPE_API_KEY'),
            )
        content = resp.output['embeddings']
        vectors = np.array([item['embedding'] for item in content])
        if normalize_embeddings:
            vectors = vectors/np.sqrt(np.sum(vectors**2, axis=1, keepdims=True))
        return vectors


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
    if model_name == 'qwen':
        return QwenEmbedding()
    model = SentenceTransformer(str(sentence_transformer_path/model_name),device=device)
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
