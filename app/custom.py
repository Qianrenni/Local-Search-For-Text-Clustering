import numpy as np
import time
from typing import Optional
from app.util import sample, get_labels, cost

class KMeans:
    """
    标准 K-Means 聚类算法实现
    """
    def __init__(self, n_clusters: int = 8, max_iter: int = 300, tol: float = 1e-4, random_state: Optional[int] = None):
        """
        Args:
            n_clusters: 聚类数量
            max_iter: 最大迭代次数
            tol: 收敛容差 (中心点移动的距离阈值)
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # 模型属性
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None  # 即 cost
        self.n_iter_ = 0

    def _init_centers(self, x: np.ndarray) -> np.ndarray:
        """
        K-Means++ 初始化或随机初始化
        这里使用简单的随机采样初始化，如需 K-Means++ 可扩展
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = x.shape[0]
        # 随机选择 n_clusters 个不重复的索引
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return x[indices].copy()

    def fit(self, x: np.ndarray) -> 'KMeansCustom':
        """
        训练模型
        Args:
            x: 输入数据 [n_samples, n_features]
        Returns:
            self
        """
        n_samples, n_features = x.shape
        
        # 1. 初始化中心点
        centers = self._init_centers(x)
        
        for i in range(self.max_iter):
            # 2. 分配标签 (E-step)
            # 使用提供的工具函数
            labels = get_labels(x, centers)
            
            # 3. 更新中心点 (M-step)
            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                cluster_points = x[labels == k]
                if len(cluster_points) > 0:
                    new_centers[k] = cluster_points.mean(axis=0)
                else:
                    # 如果簇为空，保持原中心或重新随机初始化 (这里保持原中心)
                    new_centers[k] = centers[k]
            
            # 4. 检查收敛
            # 计算中心点的移动距离 (L2 norm)
            center_shift = np.sqrt(np.sum((new_centers - centers) ** 2))
            
            # 更新状态
            centers = new_centers
            
            if center_shift < self.tol:
                break
                
        # 保存最终结果
        self.cluster_centers_ = centers
        self.labels_ = get_labels(x, centers)
        self.inertia_ = cost(x, centers)
        self.n_iter_ = i + 1
        
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测样本所属的簇
        """
        if self.cluster_centers_ is None:
            raise Exception("Model has not been fitted yet.")
        return get_labels(x, self.cluster_centers_)


class MiniBatchKMeans:
    """
    Mini-Batch K-Means 聚类算法实现
    """
    def __init__(self, n_clusters: int = 8, batch_size: int = 1024, max_iter: int = 100, 
                 tol: float = 1e-4, random_state: Optional[int] = None):
        """
        Args:
            n_clusters: 聚类数量
            batch_size: 每个 mini-batch 的样本数
            max_iter: 最大迭代次数 (指 batch 更新的次数)
            tol: 收敛容差 (基于中心点变化的平滑值)
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # 模型属性
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _init_centers(self, x: np.ndarray) -> np.ndarray:
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples = x.shape[0]
        # 初始中心点可以从第一个 batch 中选取，或者全局随机
        # 这里采用全局随机采样，通常效果更好
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return x[indices].copy()

    def fit(self, x: np.ndarray) -> 'MiniBatchKMeansCustom':
        """
        训练模型
        """
        n_samples, n_features = x.shape
        
        # 1. 初始化中心点
        centers = self._init_centers(x)
        
        # 用于记录每个中心点被分配到的样本总数，用于计算学习率
        counts = np.zeros(self.n_clusters, dtype=np.int64)
        
        old_centers = centers.copy()
        
        for i in range(self.max_iter):
            # 2. 采样 Batch
            # 使用提供的 sample 工具
            batch_points = sample(x, self.batch_size)
            
            # 3. 分配 Batch 中的样本到最近的中心
            batch_labels = get_labels(batch_points, centers)
            
            # 4. 更新中心点 (使用增量更新公式)
            # 对于每个簇 k，新中心 = (旧中心 * 旧计数 + 新样本均值 * 新样本数) / (旧计数 + 新样本数)
            # 或者更简单的 EMA: center = center * (1 - lr) + batch_mean * lr
            
            for k in range(self.n_clusters):
                # 找出当前 batch 中属于簇 k 的点
                mask = batch_labels == k
                if np.any(mask):
                    batch_cluster_points = batch_points[mask]
                    # 增加计数
                    counts[k] += len(batch_cluster_points)
                    
                    # 计算学习率: 1 / 该簇目前看到的总样本数
                    lr = 1.0 / counts[k]
                    
                    # 计算当前 batch 中该簇的均值
                    batch_mean = batch_cluster_points.mean(axis=0)
                    
                    # 增量更新中心点
                    centers[k] = (1 - lr) * centers[k] + lr * batch_mean

            # 5. 检查收敛 (可选，每 N 次迭代检查一次以节省开销)

            center_shift = np.sqrt(np.sum((centers - old_centers) ** 2))
            old_centers = centers.copy()
            
            if center_shift < self.tol:
                break
                    
        # 保存最终结果
        self.cluster_centers_ = centers
        # 注意：MBKM 的 labels_ 通常是对全量数据进行一次最终预测得到的
        self.labels_ = get_labels(x, centers)
        self.inertia_ = cost(x, centers)
        self.n_iter_ = i + 1
        
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise Exception("Model has not been fitted yet.")
        return get_labels(x, self.cluster_centers_)


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    # 生成模拟数据
    from sklearn.datasets import make_blobs
    
    print("Generating data...")
    X, y_true = make_blobs(n_samples=5000, centers=5, n_features=10, random_state=42)
    
    # --- 测试 K-Means ---
    print("\n--- Testing Custom K-Means ---")
    km = KMeansCustom(n_clusters=5, max_iter=100, random_state=42)
    
    start_time = time.time()
    km.fit(X)
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Iterations: {km.n_iter_}")
    print(f"Inertia (Cost): {km.inertia_:.4f}")
    print(f"Centers shape: {km.cluster_centers_.shape}")
    
    # --- 测试 Mini-Batch K-Means ---
    print("\n--- Testing Custom Mini-Batch K-Means ---")
    mbkm = MiniBatchKMeansCustom(n_clusters=5, batch_size=256, max_iter=100, random_state=42)
    
    start_time = time.time()
    mbkm.fit(X)
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Iterations: {mbkm.n_iter_}")
    print(f"Inertia (Cost): {mbkm.inertia_:.4f}")
    print(f"Centers shape: {mbkm.cluster_centers_.shape}")
    
    # 简单对比 sklearn (如果已安装)
    try:
        from sklearn.cluster import KMeans, MiniBatchKMeans
        print("\n--- Comparing with Sklearn (Reference) ---")
        
        sk_km = KMeans(n_clusters=5, n_init=1, random_state=42).fit(X)
        print(f"Sklearn KMeans Inertia: {sk_km.inertia_:.4f}")
        
        sk_mbkm = MiniBatchKMeans(n_clusters=5, batch_size=256, n_init=1, random_state=42).fit(X)
        print(f"Sklearn MBKMeans Inertia: {sk_mbkm.inertia_:.4f}")
        
    except ImportError:
        print("\nSklearn not installed, skipping reference comparison.")