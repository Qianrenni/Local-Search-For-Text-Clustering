import numpy as np
import torch

def l2_distance(
        x: np.ndarray,
        centers: np.ndarray
):
    """
    Args:
        x (n_samples, n_features): 输入数据
        centers (n_clusters, n_features): 中心点
    Returns:
        distance (n_samples, n_clusters): 每个样本到每个中心点的距离
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if centers.ndim == 1:
        centers = centers.reshape(1, -1)
    # [n_samples,1]
    x_2 = np.sum(x ** 2, axis=1, keepdims=True)
    # [n_clusters,1]
    centers_2 = np.sum(centers ** 2, axis=1, keepdims=True)
    result = x_2 + centers_2.T - 2 * x @ centers.T
    result=np.maximum(result, 0)
    return result.squeeze()


def sample(
    sequence: np.ndarray,
    size: int,
):
    """
    从sequence中随机采样size个元素
    Args:
        sequence (np.ndarray): 输入序列
        size (int): 采样数量
    Returns:
        np.ndarray: 采样结果
    """
    if size > len(sequence):
        indices = np.random.choice(sequence.shape[0], size=size, replace=True)
        result = sequence[indices]
        return result.squeeze()
    indices = np.random.choice(sequence.shape[0],size=size,replace=False)
    result =sequence[indices]
    return result.squeeze()


def get_labels(x, centers):
    """
    计算x归于哪个中心点
    Args:
        x (n_samples, n_features): 输入数据
        centers (n_clusters, n_features): 中心点
    return labels (n_samples,)
    """
    distance = l2_distance(x, centers)
    return np.argmin(distance, axis=1)
def get_centers(x, labels, n_clusters):
    """
    计算中心点
    Args:
        x (n_samples, n_features): 输入数据
        labels (n_samples,): 标签
        n_clusters (int): 中心点数量
    Returns:
        centers (n_clusters, n_features): 中心点
    """
    centers = np.zeros((n_clusters, x.shape[1]))
    np.add.at(centers, labels, x)
    counts = np.bincount(labels,minlength=n_clusters)
    centers /= counts[:, None]
    return centers

def cost(x, centers):
    """
    计算x到centers的最小距离平方和

    Args:
        x (n_samples, n_features): 输入数据
        centers (n_clusters, n_features): 中心点
    Returns:
        float: 最小距离平方和
    """
    distance = l2_distance(x, centers)
    min_distance = np.min(distance, axis=1)
    return min_distance.sum()


def k_nearest_neighbors(x, centers, k=1):
    """
    计算x到centers的k近邻距离平方和

    Args:
        x (n_samples, n_features): 输入数据
        centers (n_clusters, n_features): 中心点
        k: 近邻数量
    Returns:
        dist (n_samples, k): k近邻距离
        indices (n_samples, k): k近邻索引
    """
    distance = torch.tensor(l2_distance(x, centers))
    dist, indices = torch.topk(distance, k, dim=1, largest=False)
    return dist.numpy(), indices.numpy()