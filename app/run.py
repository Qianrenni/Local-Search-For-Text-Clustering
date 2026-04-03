from config import SETTING
import numpy as np
import torch
import random
import math
import json

import pandas as pd
from datetime import datetime
from app.local_search import LocalSearch
from app.eval import ClusterEvaluator
from app.util import cost,sample,get_labels
import time
from tqdm import tqdm
def _args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='ag_news', help='Dataset name')
    parser.add_argument('-i','--iteration', type=int, default=30, help='Number of iterations')
    parser.add_argument('-r', '--rounds', type=int, default=0, help='Number of rounds')
    parser.add_argument('-t', '--trans', type=int, default=0, help='Number of transformations')
    parser.add_argument('-b', '--batch', type=int, default=0, help='Batch size')
    parser.add_argument('-tb', '--total_batch', type=int, default=0, help='Total batch size')
    parser.add_argument('-mbr', '--minibatch_rounds', type=int, default=0, help='Minibatch rounds')
    args = parser.parse_args()
    return args
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
                'trans',
                'batch',
                'total_batch',
                'minibatch_rounds',
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
                'trans', 
                'batch', 
                'total_batch',
                'minibatch_rounds',
                'iteration'
            ],
            inplace=True
        )
    result_dir = SETTING.RESULT / dataset_name/'local_search'
    result_dir.mkdir(parents=True, exist_ok=True)
    file_name = f'{datetime.now().strftime("%Y_%m_%d_%H_%M")}.xlsx'

    for model in dataset_dir.iterdir():
        model_name = model.name
        print(f'Run on Model {model_name}')
        data_path = dataset_dir/f'{model_name}'/ f'unnormlized_embedding.npy'
        y_path = dataset_dir/f'{model_name}' / f'y.npy'
        data = np.load(data_path)
        y = np.load(y_path)
        labels = json.load(open( dataset_dir/f'{model_name}'/ f'labels.json', 'r'))
        k = len(labels)
        dataset = f'{dataset_name}{data.shape}'
        print(f'Running on dataset(unnormalized): {dataset}')
        data_size = data.shape[0]
        rounds = math.ceil(k * 10) if args.rounds == 0 else args.rounds
        trans = math.ceil(math.sqrt(k * 10)) if args.trans == 0 else args.trans
        batch = math.ceil(32*k) if args.batch == 0 else args.batch
        total_batch = 15 if args.total_batch == 0 else args.total_batch
        minibatch_rounds = k*10 if args.minibatch_rounds == 0 else args.minibatch_rounds
        print(
            f'params:\n'
            f'  clusters: {k}\n'
            f'  rounds: {rounds}\n'
            f'  trans: {trans}\n'
            f'  batch: {batch}\n'
            f'  total_batch: {total_batch}\n'
            f'  minibatch_rounds: {minibatch_rounds}\n'
        )
        local_search = LocalSearch(
            n_clusters=k,
            rounds=rounds,
            trans=trans,
            batch=batch,
            total_batch=total_batch,
            minibatchround=minibatch_rounds
        )
        for index in tqdm(range(iteration), desc=f'iteration'):
            start_time = time.time()
            centers = sample(data, k)
            centers = local_search.local_search_bandit(data, centers)
            total_time = time.time() - start_time
            loss = cost(data, centers)
            labels = get_labels(data, centers)
            ari, nmi, acc, f1s, rs, ps = ClusterEvaluator.external_metrics(y, labels)
            ch, db = ClusterEvaluator.internal_metrics(data, labels)
            result.loc[(dataset, model_name, 'unnormlized', k, rounds, trans, batch, total_batch, minibatch_rounds, index)] = [ari, nmi, db, ch, acc, f1s, rs, ps,loss, total_time]
            result.to_excel(result_dir / file_name)
        data = np.load(dataset_dir/f'{model_name}' / f'norm_embedding.npy')
        print(f'Running on dataset(normalized): {dataset}')
        for index in tqdm(range(iteration), desc=f'iteration'):
            start_time = time.time()
            centers = sample(data, k)
            centers = local_search.local_search_bandit(data, centers)
            total_time = time.time() - start_time
            loss = cost(data, centers)
            labels = get_labels(data, centers)
            ari, nmi, acc, f1s, rs, ps = ClusterEvaluator.external_metrics(y, labels)
            ch, db = ClusterEvaluator.internal_metrics(data, labels)
            result.loc[(dataset, model_name, 'normalized', k, rounds, trans, batch, total_batch, minibatch_rounds, index)] = [ari, nmi, db, ch, acc, f1s, rs, ps,loss, total_time]
            result.to_excel(result_dir / file_name)
