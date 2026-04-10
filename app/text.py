import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from config import SETTING
from app.sentence_transformer import get_sentence_transformer
import torch
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
    batch_size:int=64,
    normlize:bool=False
):
    """
    输出文本嵌入

    Args:
        model_name (str): 模型名称
        dataset_name (str): 数据集名称
        batch_size (int): 批处理大小
    """
    train, labels = get_text_cluster_data(dataset_name)
    result_dir = SETTING.PROCESS_DATA/f'{dataset_name}'/f'{model_name}'
    if result_dir.exists():
        print(f'{result_dir} already exists')
        return 
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_sentence_transformer(model_name=model_name,device=device)
    vectors = []
    y = []
    for i in tqdm(range(0, len(train), batch_size),desc=f'Encoding {dataset_name}'):
        batch_x = train['text'][i:i+batch_size].tolist()
        batch_y = train['y'][i:i+batch_size].tolist()
        batch_embedding = model.encode(batch_x,normalize_embeddings=normlize)
        vectors.extend(batch_embedding)
        y.extend(batch_y)
    if normlize:
        assert np.isclose(np.linalg.norm(vectors, axis=1), 1).all()
    np.save(result_dir/f'{ 'norm' if normlize else 'unnormlized'}_embedding.npy', np.array(vectors,dtype=np.float32))
    np.save(result_dir/'y.npy', np.array(y))
    with open(result_dir/'labels.json', 'w') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)
def output_embedding_mean(
    model_name:str,
    dataset_name:str,
    batch_size:int=64,
    normlize:bool=False
):
    """
    输出文本嵌入

    Args:
        model_name (str): 模型名称
        dataset_name (str): 数据集名称
        batch_size (int): 批处理大小
    """
    train, labels = get_text_cluster_data(dataset_name)
    result_dir = SETTING.PROCESS_DATA/f'{dataset_name}'/f'{model_name}_mean'
    if result_dir.exists():
        print(f'{result_dir} already exists')
        return 
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_sentence_transformer(model_name=model_name,device=device)
    max_length=model.max_seq_length-2
    tokenizer = model.tokenizer
    dim = model.get_sentence_embedding_dimension()
    vectors = []
    y = []
    for i in tqdm(range(0, len(train), batch_size),desc=f'Encoding {dataset_name}'):
        temp = train['text'][i:i+batch_size].tolist()
        indices = []
        batch_x =[]
        for index,text in enumerate(temp):
            token_ids = tokenizer.encode(text,add_special_tokens=False)
            for j in range(0, len(token_ids), max_length):
                decoded_text = tokenizer.decode(token_ids[j:j+max_length],skip_special_tokens=True)
                batch_x.append(decoded_text)
                indices.append(index)
        indices = np.array(indices)
        batch_y = train['y'][i:i+batch_size].tolist()
        batch_embedding = np.zeros((len(temp), dim))
        temp_embedding = model.encode(batch_x,normalize_embeddings=normlize)
        np.add.at(batch_embedding, indices, temp_embedding)
        counts = np.bincount(indices)
        batch_embedding = batch_embedding / counts[:, None]
        vectors.extend(batch_embedding)
        y.extend(batch_y)
    np.save(result_dir/f'{ 'norm' if normlize else 'unnormlized'}_embedding.npy', np.array(vectors,dtype=np.float32))
    np.save(result_dir/'y.npy', np.array(y))
    with open(result_dir/'labels.json', 'w') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

def _args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d','--dataset_name', type=str, default='ag_news')
    parser.add_argument('-b','--batch_size', type=int, default=64)
    parser.add_argument('-m','--model',type=str,default='all-MiniLM-L6-v2')
    parser.add_argument('-t','--truncated',type=int,default=1)
    parser.add_argument('-n','--norm',type=int,default=0)
    parser.add_argument('-a','--all',type=int,default=0)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = _args()
    if args.all==1:
        model_names = [ sub_dir.name for sub_dir in SETTING.DEPENDENCY.iterdir()]
        for model_name in model_names:
            output_embedding(
                model_name=model_name,
                dataset_name=args.dataset_name,
                batch_size=args.batch_size,
                normlize=args.norm==1
            )
    else:
        if args.truncated==1:
            output_embedding(
                model_name=args.model,
                dataset_name=args.dataset_name,
                batch_size=args.batch_size,
                normlize=args.norm==1
            )
        else:
            output_embedding_mean(
                model_name=args.model,
                dataset_name=args.dataset_name,
                batch_size=args.batch_size,
                normlize=args.norm==1
            )
