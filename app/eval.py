from sklearn.metrics import (
    calinski_harabasz_score, 
    accuracy_score,
    davies_bouldin_score,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score,confusion_matrix
import numpy as np
from scipy.optimize import linear_sum_assignment

class ClusterEvaluator:
    """聚类评估"""

    @staticmethod
    def map_cluster_labels(y_true, y_pred):
        """
        使用匈牙利算法找到最优标签映射
        Args:
            y_true (np.ndarray): 真实标签, 形状为 (n_samples,)
        Returns:
            np.ndarray: 映射后的预测标签, 形状为 (n_samples,)
            dict: 标签映射字典, 键为预测标签，值为真实标签
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # 构建混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 匈牙利算法找最优匹配（最大化对角线和）
        row_ind, col_ind = linear_sum_assignment(-cm)
        
        # 创建映射字典：聚类标签 → 真实标签
        label_map = {pred: true for true, pred in zip(row_ind, col_ind)}
        
        # 应用映射
        y_pred_mapped = np.array([label_map.get(p, p) for p in y_pred])
        
        return y_pred_mapped, label_map
    
    @staticmethod
    def internal_metrics(x, labels):
        """
        计算内部指标
        Calinski-Harabasz (CH)	簇间离散度/簇内离散度比值	[0, +∞)	越大越好
        Davies-Bouldin (DB)	簇内距离与簇间距离的比值	[0, +∞)	越小越好

        Args:
            x (np.ndarray): 样本数据，形状为 (n_samples, n_features)
            labels (np.ndarray): 标签, 形状为 (n_samples,)
        Returns:
            float: Calinski-Harabasz指数
            float: Davies-Bouldin指数
        """
        # 3. Calinski-Harabasz指数
        if len(set(labels)) >= 2:
            ch_score = calinski_harabasz_score(x, labels)
        else:
            ch_score = np.nan
            
        # 4. Davies-Bouldin指数
        if len(set(labels)) >= 2:
            db_score = davies_bouldin_score(x, labels)
        else:
            db_score = np.nan
            
        return ch_score, db_score
    
    @staticmethod
    def external_metrics(labels_true, labels_pred):
        """
        计算外部指标（需要真实标签）
        Args:
            labels_true (np.ndarray): 真实标签, 形状为 (n_samples,)
            labels_pred (np.ndarray): 预测标签, 形状为 (n_samples,)
        Returns:
            float: 调整兰德指数
            float: 归一化互信息
            float: 准确率
            float: F1分数
            float: 召回率
            float: 精确率
        """
        labels_pred_mapped, _ = ClusterEvaluator.map_cluster_labels(labels_true, labels_pred)
        return \
        adjusted_rand_score(labels_true, labels_pred),\
        normalized_mutual_info_score(labels_true, labels_pred),\
        accuracy_score(labels_true, labels_pred_mapped),\
        f1_score(labels_true, labels_pred_mapped, average='weighted'),\
        recall_score(labels_true, labels_pred_mapped, average='weighted'),\
        precision_score(labels_true, labels_pred_mapped, average='weighted')