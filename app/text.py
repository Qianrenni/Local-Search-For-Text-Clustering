import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from config import SETTING
from app.sentence_transformer import get_sentence_transformer
def get_text_cluster_data(dataset_name: str):
    """
    获取文本聚类数据
    Args:
        dataset_name (str): 数据集名称
    """
    dataset_path = SETTING.DATA/f'{dataset_name}'/'train.xlsx'
    if not dataset_path.exists():
        dataset_path = SETTING.DATA/f'{dataset_name}'/'train.json'
        df = pd.read_json(dataset_path)
    else:
        df = pd.read_excel(dataset_path)
    classes = {label: i for i, label in enumerate(df['label'].unique())}
    labels = [None for _ in range(len(classes))]
    for label, i in classes.items():
        labels[i] = label if type(label) == str else int(label)
    df['y'] = df['label'].map(classes)
    return df, labels
def output_embedding(
    model_name:str,
    dataset_name:str,
    batch_size:int=64
):
    """
    输出文本嵌入

    Args:
        model_name (str): 模型名称
        dataset_name (str): 数据集名称
        batch_size (int): 批处理大小
    """
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_sentence_transformer(model_name=model_name,device=device)
    result_dir = SETTING.PROCESS_DATA/f'{dataset_name}'/f'{model_name}'
    if result_dir.exists():
        print(f'{result_dir} already exists')
        return 
    result_dir.mkdir(parents=True, exist_ok=True)
    train, labels = get_text_cluster_data(dataset_name)
    vectors = []
    y = []
    norm_vectors = []
    for i in tqdm(range(0, len(train), batch_size),desc=f'Encoding {dataset_name}'):
        batch_x = train['text'][i:i+batch_size].tolist()
        batch_y = train['y'][i:i+batch_size].tolist()
        batch_embedding = model.encode(batch_x)
        vectors.extend(batch_embedding)
        norm_embedding = batch_embedding / np.sqrt(np.sum(batch_embedding**2, axis=1, keepdims=True))
        norm_vectors.extend(norm_embedding)
        y.extend(batch_y)
    assert np.isclose(np.linalg.norm(norm_vectors, axis=1), 1).all()
    np.save(result_dir/'unnormlized_embedding.npy', np.array(vectors))
    np.save(result_dir/'norm_embedding.npy', np.array(norm_vectors))
    np.save(result_dir/'y.npy', np.array(y))
    with open(result_dir/'labels.json', 'w') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

def _args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d','--dataset_name', type=str, default='ag_news')
    parser.add_argument('-b','--batch_size', type=int, default=64)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = _args()
    model_names = [
            'all-MiniLM-L12-v2',
            'all-MiniLM-L6-v2',
            'all-mpnet-base-v2',
            'clip-ViT-B-32-multilingual-v1',
            'paraphrase-multilingual-MiniLM-L12-v2'
            ]
    for model_name in model_names:
        output_embedding(
            model_name=model_name,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size
        )
