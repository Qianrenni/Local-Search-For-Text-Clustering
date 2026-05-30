import numpy as np

from app.util import k_nearest_neighbors, l2_distance, sample


class PaperMiniBatchKMeans:
    """
    严格遵循 Schwartzman (2023) "Mini-batch k-means terminates within O(d/ε) iterations"
    论文定义的算法实现。
    理论依据:Algorithm 1 与 Section 4 Theorem 14。
    """

    def __init__(
        self,
        n_clusters: int | None = None,
        batch_size: int | None = None,
        epsilon: float | None = None,
        max_iter: int | None = None,
        random_state: int | None = None,
    ):
        """
        Args:
            n_clusters: 聚类数量
            batch_size: 批次大小
            epsilon: 收敛阈值
            max_iter: 最大迭代次数
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_iter_ = 0
        # 目标函数值
        self.inertia_ = None

    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        return sample(X, self.n_clusters)

    def fit(self, X: np.ndarray):
        centers = self._init_centers(X)
        for i in range(self.max_iter):
             # 1. 均匀随机采样批次（允许重复采样，符合论文假设）
            batch = sample(X, self.batch_size, is_replace=True)

            # 2. 计算更新前的批次目标函数值 f_B(C_i)

            min_dist_before,labels_batch = k_nearest_neighbors(batch,centers)
            cost_before = np.mean(min_dist_before)
            # 4. 依据论文定理14的学习率与更新规则计算新中心
            new_centers = centers.copy()
            labels_batch = labels_batch.squeeze()
            for j in range(self.n_clusters):
                mask = labels_batch == j
                b_j = np.sum(mask)
                if b_j == 0:
                    continue
                batch_mean = batch[mask].mean(axis=0)
                # 核心学习率公式 α_j = sqrt(b_j / b)，不随时间衰减
                alpha_j = np.sqrt(b_j / self.batch_size)
                new_centers[j] = (1.0 - alpha_j) * centers[j] + alpha_j * batch_mean

            # 5. 计算更新后的批次目标函数值 f_B(C_{i+1})
            dist_after = l2_distance(batch, new_centers)
            min_dist_after = np.min(dist_after, axis=1)
            cost_after = np.mean(min_dist_after)

            # 6. 论文定义的停止条件：局部改进量低于 ε 即终止
            improvement = cost_before - cost_after
            centers = new_centers
            if improvement < self.epsilon:
                break

        self.cluster_centers_ = centers
        self.n_iter_ = i + 1
        return self
