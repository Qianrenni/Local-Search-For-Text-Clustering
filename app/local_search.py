import math
import numpy as np
# from sklearn.cluster import MiniBatchKMeans
from app.util import l2_distance, sample, k_nearest_neighbors
class LocalSearch(object):
    """
    局部搜索聚类
    """

    def __init__(
            self,
            n_clusters=None,
            rounds=400,
            trans=10,
            batch=100,
            total_batch=10,
            minibatchround=40,
            threshold=0,
            epsilon=0.1,
    ):
        """
        Args:
            n_clusters: 聚类数量
            rounds: 局部搜索迭代轮数
            trans: 候选点探索步数
            batch: 每次Bandit评估时的单批采样数
            total_batch: Bandit评估总批次数
            minibatchround: 最终微调的迭代轮数
            threshold: 微调改进阈值
            epsilon: 贪婪选择概率
        """

        self.n_clusters_ = n_clusters
        self.rounds_ = rounds
        self.trans_ = trans
        self.batch_ = batch
        self.total_batch_ = total_batch
        self.minibatchround_ = minibatchround
        self.threshold_ = threshold
        self.epsilon_ = min(1, epsilon)
        
    def minibatch_kmeans(
        self,
        data,
        centers,
        batch,
        rounds,
        threshold=0
    ):
        """
        Mini-batch K-means算法
        Args:
            data (n_samples, n_features): 输入数据
            centers (n_clusters, n_features): 初始中心点
            batch (int): 每次迭代的样本数量
            rounds (int): 迭代轮数
            threshold (float): 容差
        Returns:
            centers (n_clusters, n_features): 最终中心点
        """
        # cost改进比例
        ratio = 1
        center_nums = centers.shape[0]
        for j in range(rounds):
            # 如果占比大于1+threshold，则增加batch
            if (j > 0) and (ratio > (1 + threshold)):
                batch = math.ceil(batch * ratio)

            # 随机选择 batch 个样本
            points = sample(data, batch)
            # distance [n_samples, 1]
            # indexes [n_samples, 1]
            # 找到归属类族的索引
            distance, indexes = k_nearest_neighbors(points, centers)
            distance = distance[:, 0]
            indexes = indexes[:, 0]
            # 旧中心点的cost
            cost_old_centers = np.sum(distance)
            # 计算新的中心点
            centers_now = np.zeros_like(centers)
            # 统计每个中心点的样本数
            count = np.bincount(indexes, minlength=center_nums)
            np.add.at(centers_now, indexes, points)

            nonzero = count > 0
            centers_now[nonzero] /= count[nonzero][:, None]
            centers_now[~nonzero] = centers[~nonzero]

            distance_new = l2_distance(points, centers_now)
            # 每个样本到最近中心点的距离平方
            distance_new = np.min(distance_new, axis=1)
            # 新中心点的cost
            cost_new_centers = distance_new.sum()
            new_ratio = cost_old_centers / cost_new_centers
            # 收敛判断：如果改进很小就提前返回
            if new_ratio < (1 + threshold):
                return centers_now if (new_ratio > 1) else centers
            if new_ratio > 1:
                # 更新改进比例
                ratio = new_ratio
                centers = centers_now
        return centers

    def local_search_bandit(self, x, centers):
        """
        局部搜索聚类
        Args:
            x (n_samples, n_features): 输入数据
            centers (n_clusters, n_features): 初始中心点
        Returns:
            centers (n_clusters, n_features): 最终中心点
        """
        sample_size = x.shape[0]
        points = sample(x,self.rounds_)
        for i in range(self.rounds_):
            # 获取下一个样本点
            next_point = points[i]
            # 计算样本点到中心点的距离平方
            x_distance_with_centers = l2_distance(next_point, centers)
            # 找到最小距离
            x_min_distance = np.min(x_distance_with_centers)
            # 如果最小距离为0，则跳过
            if x_min_distance == 0:
                continue
            trans_points = sample(x, self.trans_)
            trans_points_distance_with_centers = l2_distance(trans_points, centers)
            trans_min_distance = np.min(trans_points_distance_with_centers, axis=1)
            for step in range(self.trans_):
                y_point = trans_points[step]
                y_min_distance = trans_min_distance[step]
                if x_min_distance < 1e-9:
                    x_min_distance = 0.01
                    continue
                ratio = y_min_distance/x_min_distance
                if ratio > np.random.rand():
                    x_min_distance = y_min_distance
                    x_distance_with_centers = trans_points_distance_with_centers[step]
                    next_point = y_point

            next_center_index = np.argmin(x_distance_with_centers)
            exclude_mask = np.ones(self.n_clusters_, dtype=bool)
            exclude_mask[next_center_index] = False
            next_random_center_index = np.random.choice(np.arange(self.n_clusters_)[exclude_mask])
            # 已经被使用的数量
            used_count = 0
            # 总共需要使用的数量
            total_counts = self.batch_ * self.total_batch_
            solution = [next_center_index, next_random_center_index]
            s_mean = np.zeros(self.n_clusters_)
            s_std = np.zeros(self.n_clusters_)
            std_history = np.zeros(self.n_clusters_)
            flag = 0
            delta = 1 / sample_size
            while (used_count < total_counts and len(solution) > 1):
                solution_np = np.array(solution, dtype=int)
                batch_points = sample(x, self.batch_)
                # [batch_size, nearest_neighbors_size], [batch_size, nearest_neighbors_size]
                distance, k_center_indexes = k_nearest_neighbors(batch_points, centers, 2)
                # 选择最近的中心点
                k_center_indexes = k_center_indexes[:, 0]

                for step in range(len(solution)):
                    current_center_index = solution[step]
                    # 计算样本点到下一个中心点的距离平方
                    batch_dis_with_next_point = l2_distance(batch_points, next_point)
                    # 样本点到最近中心的距离
                    distance_nearest = distance.copy()[:, 0]
                    # 归于当前step索引的中心的样本
                    points_index_affected = k_center_indexes == solution[step]
                    # 将其距离更新为第二近的距离
                    distance_nearest[points_index_affected] = (distance[points_index_affected])[:, 1]
                    # 计算成本变化:新距离-旧距离
                    distance_difference = batch_dis_with_next_point - distance_nearest
                    # 有增益的样本点
                    index_large = distance_difference < 0
                    # 更新距离为新距离
                    distance_nearest[index_large] = batch_dis_with_next_point[index_large]
                    # 计算成本变化:新距离-旧距离
                    distance_difference = distance_nearest - distance[:, 0]
                    # 更新统计量
                    if flag == 0:  # 第一批样本
                        temp_std = distance_difference.std()
                        s_mean[current_center_index] = distance_difference.mean()
                        std_history[current_center_index] = (temp_std ** 2) * self.batch_
                        s_std[current_center_index] = temp_std * np.sqrt(2 * np.log10(sample_size) / self.batch_)
                    else:

                        smean_old = s_mean[current_center_index]
                        s_mean[current_center_index] = (used_count * s_mean[
                            current_center_index] + distance_difference.sum()) / (used_count + self.batch_)

                        std_sum = std_history[current_center_index] + (
                                (distance_difference - smean_old) * (distance_difference - s_mean[current_center_index])
                        ).sum()
                        std = np.sqrt(std_sum / (used_count + self.batch_))
                        std_history[current_center_index] = std_sum
                        s_std[current_center_index] = std*self.epsilon_*np.sqrt(np.log10(sample_size) / (self.batch_ + used_count))
                flag = 1
                used_count += self.batch_
                # UCB淘汰策略：保留"可能最优"的交换对
                target_min = np.min(s_mean[solution_np]+s_std[solution_np])

                target_min = min(target_min, 0)

                target_diff = s_mean[solution_np] - s_std[solution_np]  # UCB下界

                # 淘汰：如果一个交换对的下界 > 其他交换对的上界，则它肯定不是最优
                solution = solution_np[target_diff <= target_min].tolist()

            if target_min == 0:
                continue

            if len(solution) == 0:
                continue
            if len(solution) == 1:
                centers[solution[0]] = next_point
        # min_kmeans = MiniBatchKMeans(
        #     n_clusters=self.n_clusters_,
        #     init=centers,
        #     n_init=1,
        #     max_iter=self.minibatchround_,
        #     batch_size=self.batch_,
        #     compute_labels=False,
        #     tol=0.05
        # )
        # return min_kmeans.fit(x).cluster_centers_
        return self.minibatch_kmeans(x, centers, self.batch_, self.minibatchround_, self.threshold_)
        # return centers

    def fast_local_search(self, x, centers):
        """
        Args:
            x (sample_size, n_features)
            centers: (n_clusters, n_features)
        """
        # Strategy 1: Use more space comlexity and heap structure for fast local search implementation"
        # Preprocessing steps for storing the distance and assignment information"
        INF = float('inf')
        sample_size = x.shape[0]
        center_size = centers.shape[0]
        # [最近距离,第二近距离]
        dist_2 = np.ones([sample_size, 2])
        dist = (l2_distance(x, centers))
        sort_indexes = np.argsort(dist, axis=1)
        affected_list = [[] for i in range(center_size)]
        for i in range(sample_size):
            dist_2[i][0] = dist[i][sort_indexes[i][0]]
            dist_2[i][1] = dist[i][sort_indexes[i][1]]
            # 记录每个族的归属
            affected_list[sort_indexes[i][0]].append(i)

        # "Calculating the current clustering cost"
        cost_now = (dist_2[:, 0]).sum()

        # "Construct the sampling distribution"
        # 按照最近距离的大小进行采样
        prob_modified = dist_2[:, 0] / (dist_2[:, 0].sum())

        sample_list = list(range(sample_size))
        # "Start the Local Search Process"
        candidate_next = np.random.choice(sample_list, size=1, p=prob_modified).tolist()
        for i in range(0, self.rounds_):

            cost_min = INF
            swap_id = None

            # "Find the oversampling factor"

            # "Construct the sampling distribution"

            # "Sample one data point from the modified probability"
            next_point = candidate_next[i]
            centers_new = (x[next_point]).reshape(1, -1)
            dist_tot_new = (l2_distance(x, centers_new))[:, 0]

            # "Make the comparison between the distances of nearest and the new centers"
            dist_tot_new_modified = (dist_2.copy())[:, 0]
            dist_diff = dist_tot_new_modified - dist_tot_new
            dist_large_id = (np.argwhere(dist_diff > 0))[:, 0]
            dist_tot_new_modified[dist_large_id] = dist_tot_new[dist_large_id]

            # "Finding the minimum id"
            next_numpy = data[next_point]
            next_numpy = next_numpy.reshape(1, -1)
            pd_k = l2_distance(centers, next_numpy)[:, 0]
            min_id = np.argmin(pd_k)
            rd_id = sample(range(0, len(centers)), 1)[0]

            # "Try to enumerate possible swap pairs"
            for j in [min_id, rd_id]:
                # "Find the points whose closest center are swapped out"
                # "Now try to swap the j-th center out"
                dist_temp = dist_2.copy()
                id_affected = np.array(affected_list[j], dtype=int)

                # "Compare the distances and calculate the new cost"
                dist_affected_modified = (dist_temp[id_affected])[:, 1]
                pd = dist_tot_new[id_affected]
                dist_diff = dist_affected_modified - pd
                id_large = (np.argwhere(dist_diff > 0))[:, 0]
                dist_affected_modified[id_large] = pd[id_large]
                cost_new = dist_tot_new_modified.sum() - (
                    dist_tot_new_modified[id_affected]).sum() + dist_affected_modified.sum()

                # "Judge if the swap is feasible"
                if (cost_new < cost_min):
                    cost_min = cost_new
                    swap_id = j

            # "Check whether the minimum cost swap is feasible"
            if (cost_min < (1 - 1 / (100 * self.n_clusters_)) * cost_now):

                # "Perform this swap"
                cost_now = cost_min
                centers[swap_id] = x[next_point]
                # "Renew the distance structures"
                center_new = (x[next_point]).reshape(1, -1)
                pd = (l2_distance(x, center_new))[:, 0]
                dist[:, swap_id] = pd
                sort_indexes = np.argsort(dist, axis=1)
                dist_2 = np.ones([x.shape[0], 2])
                affected_list = [[] for j1 in range(0, centers.shape[0])]
                for j1 in range(0, x.shape[0]):
                    dist_2[j1][0] = dist[j1][sort_indexes[j1][0]]
                    dist_2[j1][1] = dist[j1][sort_indexes[j1][1]]
                    affected_list[sort_indexes[j1][0]].append(j1)

                prob_modified = dist_2[:, 0] / (dist_2[:, 0].sum())
            else:
                continue
        mini_batch_size = math.ceil(0.01 * data.shape[0])
        mini_batch_size = min(math.ceil(data.shape[0] * 0.1), mini_batch_size)
        centers = self.minibatch_kmeans(x, centers, mini_batch_size, self.minibatchround_, 0.05)
        return centers
