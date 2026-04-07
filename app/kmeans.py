from pathlib import Path

from config import SETTING
import numpy as np
import torch
import random
import math
import json

import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
# from app.custom import KMeans
from app.eval import ClusterEvaluator
from app.util import cost,get_labels
import time
from tqdm import tqdm
def _args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='ag_news', help='Dataset name')
    parser.add_argument('-i','--iteration', type=int, default=30, help='Number of iterations')
    parser.add_argument('-r', '--rounds', type=int, default=0, help='Number of rounds')
    parser.add_argument('-t','--tol', type=float, default=0.001, help='Tolerance for convergence')
    parser.add_argument('-m','--model',type=str,default='all-MiniLM-L6-v2')
    parser.add_argument('-a','--all',type=int,default=0)
    parser.add_argument('-n','--norm',type=int,default=1)
    parser.add_argument('-k','--clusters',type=int,default=-1)
    args = parser.parse_args()
    return args

def run(
    model_name:str,
    dataset_name:str,
    data_path:Path,
    y:np.ndarray,
    args,
    labels,
    result:pd.DataFrame,
    save_path:Path
):
    data = np.load(data_path)
    k = len(labels) if args.clusters==-1 else args.clusters 
    dataset = f'{dataset_name}{data.shape}'
    data_size = data.shape[0]
    rounds = math.ceil(128*15*15*k*k/data_size) if args.rounds == 0 else args.rounds
    print(
        f'params:\n'
        f'  data_size{data.shape}\n'
        f'  clusters: {k}\n'
        f'  rounds: {rounds}\n'
        f'  tolerance: {args.tol}\n'
    )
    kmeans = KMeans(
        n_clusters=k,
        max_iter=rounds,
        tol=args.tol,
    )
    for index in tqdm(range(iteration), desc=f'iteration'):
        start_time = time.time()
        kmeans.fit(data)
        centers = kmeans.cluster_centers_
        total_time = time.time() - start_time
        loss = cost(data, centers)
        labels = get_labels(data, centers)
        ari, nmi, acc, f1s, rs, ps = ClusterEvaluator.external_metrics(y, labels)
        ch, db = ClusterEvaluator.internal_metrics(data, labels)
        result.loc[(dataset, model_name, data_path.name.split('_')[0], k, rounds, args.tol, index)] = [ari, nmi, db, ch, acc, f1s, rs, ps,loss, total_time]
        result.to_excel(save_path)
if __name__ == '__main__':
    np.random.seed(SETTING.SEED)
    random.seed(SETTING.SEED)
    torch.manual_seed(SETTING.SEED)
    torch.cuda.manual_seed(SETTING.SEED)
    args = _args()
    dataset_name:str = args.dataset
    dataset_dir =SETTING.PROCESS_DATA / f'{dataset_name}' 
    iteration = args.iteration
    result = pd.DataFrame(
            columns=[
                'dataset',
                'model',
                'norm',
                'clusters',
                'rounds',
                'tol',
                'iteration',
                'ARI',
                'NMI',
                'DB',
                'CH',
                'ACC',
                'F1S',
                'RS',
                'PS',
                'cost', 
                'time'
            ]
        )
    result.set_index(
            [
                'dataset',
                'model' ,
                'norm', 
                'clusters', 
                'rounds', 
                'tol',
                'iteration'
            ],
            inplace=True
        )
    result_dir = SETTING.RESULT / dataset_name/'kmeans'
    result_dir.mkdir(parents=True, exist_ok=True)
    file_name = f'{datetime.now().strftime("%Y_%m_%d_%H_%M")}.xlsx'

    if args.all==1:
        for model in dataset_dir.iterdir():
            model_name = model.name
            labels = json.loads((model/'labels.json').read_text())
            y = np.load(model/'y.npy')
            norm_embedding_path = model/'norm_embedding.npy'
            if norm_embedding_path.exists():
                run(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    data_path=norm_embedding_path,
                    y=y,
                    args=args,
                    result=result,
                    save_path=result/file_name
                )
            unnormlized_embedding_path = model/'unnormlized_embedding.npy'
            if unnormlized_embedding_path.exists():
                run(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    data_path=norm_embedding_path,
                    y=y,
                    args=args,
                    result=result,
                    save_path=result/file_name
                )
    else:
        model_dir = dataset_dir/f'{args.model}'
        if model_dir.exists():
            labels = json.loads((model_dir/'labels.json').read_text())
            y = np.load(model_dir/'y.npy')
            data_path = model_dir/f'{'norm' if args.norm else 'unnormlized'}_embedding.npy'
            run(
                model_name=model_dir.name,
                dataset_name=dataset_name,
                data_path=data_path,
                y=y,
                labels=labels,
                args=args,
                result=result,
                save_path=result_dir/file_name
            )
    result.reset_index(inplace=True)
    arrgregate_df = result.groupby(['dataset',
                                    'model',
                                    'norm',
                                    'clusters',
                                    'rounds',
                                    'tol',
                                ]).agg(['mean', 'std']).reset_index()
    arrgregate_df.to_excel(result_dir / f'aggregate_{file_name}')