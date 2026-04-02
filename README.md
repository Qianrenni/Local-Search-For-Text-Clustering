# 基于局部搜索的聚类分析系统

## 项目介绍

本项目实现了一种基于**局部搜索算法**的高效聚类分析方法，结合了 Mini-batch K-means 和 Bandit 算法进行优化。项目支持多种数据集和预训练嵌入模型，提供了完整的聚类评估体系。

### 核心特性

- 🎯 **局部搜索算法**：使用局部搜索策略优化聚类中心，避免陷入局部最优
- 🚀 **Bandit 算法优化**：采用 Multi-Armed Bandit 策略高效评估候选中心点
- 📊 **Mini-batch K-means**：支持大规模数据的批量处理，提升计算效率
- 🔍 **多模型支持**：兼容 Sentence Transformer 等多种嵌入模型
- 📈 **全面评估体系**：支持 ARI、NMI、ACC、F1-Score、DB、CH 等多种评估指标
- 💾 **数据预处理**：自动处理标准化/归一化嵌入向量

### 支持的评估指标

**外部指标**（需要真实标签）：
- ARI (Adjusted Rand Index)
- NMI (Normalized Mutual Information)
- ACC (Accuracy)
- F1-Score
- Recall Score
- Precision Score

**内部指标**（无需真实标签）：
- CH (Calinski-Harabasz Score)
- DB (Davies-Bouldin Index)

---

# 快速开始

## 依赖安装  

1. 安装 Python 依赖包
    ```bash
    pip install -r ./requirements.txt
    ```

2. 下载预训练 Sentence-Transformer 模型
    ```bash
    modelscope download --model sentence-transformers/all-MiniLM-L6-v2
    ```

3. 设置环境变量
    ```powershell
    $env:PYTHONPATH = "."
    ```

## 运行示例

### 基础运行

```bash
python app/run.py -d ag_news -i 30
```

参数说明：
- `-d, --dataset`: 数据集名称（默认：ag_news）
- `-i, --iteration`: 迭代次数（默认：30）

### 支持的数据集

项目支持以下数据集（需放置在 `processed_data` 目录）：
- `ag_news` - AG News 新闻分类数据集
- `20_newsgroups` - 20 个新闻组数据集
- `arxivs2s` - arXiv 论文摘要数据集
- 其他自定义数据集（按格式放置即可）

---

## 项目结构

```
d:\graduate_design\code\
├── app/                      # 核心应用代码
│   ├── run.py               # 主程序入口
│   ├── local_search.py      # 局部搜索算法实现
│   ├── eval.py              # 聚类评估工具
│   ├── util.py              # 工具函数（距离计算、采样等）
│   ├── args.py              # 命令行参数配置
│   └── text.py              # 文本处理工具
├── data/                     # 原始数据集
│   ├── 20_newsgroups/
│   ├── ag_news/
│   ├── amazon_massive/
│   └── ...
├── processed_data/           # 预处理后的数据（嵌入向量）
│   ├── ag_news/
│   │   └── all_minilm_l6/
│   │       ├── labels.json
│   │       ├── norm_embedding.npy      # 归一化嵌入
│   │       ├── unnormlized_embedding.npy  # 未归一化嵌入
│   │       └── y.npy       # 真实标签
│   └── ...
├── result/                   # 运行结果输出
│   └── ag_news/
│       └── YYYY_MM_DD_HH_MM_local_search.xlsx
├── dependency/               # 预训练模型文件
│   └── sentence-transformers/
├── config.py                 # 配置文件
├── requirements.txt          # Python 依赖
└── README.md                 # 项目文档
```

---

## 算法流程

### 整体流程

1. **数据准备**：加载预处理的嵌入向量和真实标签
2. **初始化**：随机采样 k 个初始聚类中心
3. **局部搜索优化**（核心阶段）：
   - 在每轮迭代中探索 `trans` 个候选点
   - 使用 Bandit 算法评估候选点质量
   - 选择最优候选点更新聚类中心
4. **Mini-batch K-means 微调**：对最终中心点进行精细调整
5. **结果评估**：计算多种聚类和评估指标
6. **输出生成**：将结果保存为 Excel 文件

### 核心算法详解

#### 1. 局部搜索算法 (Local Search)

局部搜索算法通过迭代优化聚类中心，避免陷入局部最优：

```
for each round:
    1. 采样一个样本点 next_point
    2. 计算该点到所有中心的距离
    3. 探索 trans 个候选点：
       - 采样候选点 y_point
       - 使用 Bandit 策略评估是否替换
       - 保留更优的候选点
    4. 确定最优替换位置：
       - 使用 UCB (Upper Confidence Bound) 策略
       - 淘汰次优的交换对
       - 更新聚类中心
```

#### 2. Bandit 算法评估

采用 Multi-Armed Bandit 策略高效评估候选中心点：

- **置信区间估计**：计算每个候选交换的均值和标准差
- **UCB 淘汰策略**：保留"可能最优"的交换对
- **渐进式评估**：通过多批次采样逐步缩小候选范围

#### 3. Mini-batch K-means 微调

对局部搜索得到的中心点进行精细化调整：

- 动态调整批处理大小
- 基于成本改进比例控制收敛
- 支持早停机制（patience 策略）

### 算法复杂度

- **时间复杂度**：O(rounds × trans × batch × total_batch)
- **空间复杂度**：O(n_samples × n_features + n_clusters × n_features)

### 核心参数配置

#### 默认参数（自动计算）

```python
# 聚类数量
k = len(labels)  # 根据数据集标签数自动确定

# 局部搜索迭代轮数
rounds = min(k * 100, data_size * 0.2)  # k 的 100 倍或数据量的 20%

# 候选点探索步数
trans = 64  # 每轮迭代探索的候选点数量

# 批处理大小
batch = 512  # Bandit 评估时的单批采样数

# Bandit 评估总批次数
total_batch = ceil((data_size / 2) / batch)  # 总评估批次

# Mini-batch 微调轮数
minibatch_rounds = 40  # 最终微调的迭代轮数
```

#### 自定义参数

可通过命令行参数覆盖默认值：

```bash
python app/run.py \
    -d ag_news \
    -i 30 \
    -r 500 \
    -t 128 \
    -b 256 \
    -tb 20 \
    -mbr 60
```

参数说明：
- `-r, --rounds`: 局部搜索迭代轮数（默认：自动计算）
- `-t, --trans`: 候选点探索步数（默认：64）
- `-b, --batch`: 批处理大小（默认：512）
- `-tb, --total_batch`: Bandit 评估总批次数（默认：自动计算）
- `-mbr, --minibatch_rounds`: Mini-batch 微调轮数（默认：40）

#### 参数调优建议

| 场景 | rounds | trans | batch | 说明 |
|------|--------|-------|-------|------|
| 小规模数据 (<10K) | k×50 | 32 | 256 | 快速收敛 |
| 中等规模 (10K-100K) | k×100 | 64 | 512 | 平衡性能 |
| 大规模数据 (>100K) | k×150 | 128 | 1024 | 保证质量 |
| 高精度需求 | k×200 | 256 | 512 | 最优结果 |

---

## 输出结果

运行结果将保存在 `result/{dataset_name}/` 目录下，文件名为时间戳格式的 Excel 文件：

```
YYYY_MM_DD_HH_MM_local_search.xlsx
```

Excel 包含以下列：
- dataset: 数据集信息
- model: 嵌入模型名称
- norm: 是否归一化（normalized/unnormlized）
- clusters: 聚类数量
- rounds/trans/batch/minibatch_rounds: 算法参数
- iteration: 迭代索引
- ARI/NMI/DB/CH/ACC/F1S/RS/PS: 评估指标
- cost: 聚类损失函数值
- time: 运行时间（秒）

---

## 高级用法

### 1. 文本嵌入生成

如果需要使用自定义数据集或模型，可先生成文本嵌入：

```bash
# 使用默认配置生成嵌入
python app/text.py -m all_minilm_l6 -d ag_news -b 64

# 参数说明
# -m, --model_name: 模型名称（默认：all_minilm_l6）
# -d, --dataset_name: 数据集名称（默认：ag_news）
# -b, --batch_size: 批处理大小（默认：64）
```

生成的嵌入文件将保存在 `processed_data/{dataset_name}/{model_name}/` 目录下。

### 2. 多次独立运行

为了获得更稳定的结果，建议进行多次独立运行：

```bash
# 运行 50 次独立实验
python app/run.py -d ag_news -i 50
```

每次迭代会随机初始化聚类中心，最终结果取平均值。

### 3. 结果分析

运行完成后，可使用以下代码分析结果：

```python
import pandas as pd

# 读取结果
result = pd.read_excel('result/ag_news/2026_04_02_23_06_local_search.xlsx')

# 统计各指标的平均值和标准差
stats = result.groupby(['model', 'norm'])[['ARI', 'NMI', 'ACC', 'F1S']].agg(['mean', 'std'])
print(stats)

# 比较归一化和未归一化的效果
comparison = result.pivot_table(
    index=['model'], 
    columns='norm', 
    values=['ARI', 'NMI', 'ACC']
)
print(comparison)
```

### 4. 自定义数据集格式

如需添加新数据集，请按以下格式准备：

**原始数据格式**（`data/{dataset_name}/train.xlsx`）：
```python
import pandas as pd

# 包含 'text' 和 'label' 两列
df = pd.DataFrame({
    'text': ['文本内容 1', '文本内容 2', ...],
    'label': ['类别 A', '类别 B', ...]
})
df.to_excel('data/my_dataset/train.xlsx', index=False)
```

**预处理后的数据**（`processed_data/{dataset_name}/{model_name}/`）：
- `unnormlized_embedding.npy` - 原始嵌入向量 (n_samples, n_features)
- `norm_embedding.npy` - 归一化嵌入向量 (n_samples, n_features)
- `y.npy` - 真实标签 (n_samples,)
- `labels.json` - 标签名称列表 ["类别 A", "类别 B", ...]

---

## 核心代码结构

### app/run.py - 主程序入口

负责整个聚类流程的调度和控制：
- 加载数据和模型配置
- 初始化 LocalSearch 对象
- 执行多次独立迭代
- 计算评估指标并保存结果

### app/local_search.py - 局部搜索算法

核心算法实现，包含三个主要方法：

- **`local_search_bandit()`**: 基于 Bandit 算法的局部搜索
  - 实现候选点探索和评估
  - UCB 淘汰策略选择最优交换对
  
- **`minibatch_kmeans()`**: Mini-batch K-means 微调
  - 动态批处理大小调整
  - 基于成本改进的收敛判断
  
- **`fast_local_search()`**: 快速局部搜索（备用方案）
  - 使用堆结构加速查找
  - 优化的距离更新策略

### app/eval.py - 聚类评估工具

提供全面的聚类评估指标：

- **`external_metrics()`**: 外部指标（需要真实标签）
  - ARI, NMI, ACC, F1-Score, Recall, Precision
  - 使用匈牙利算法进行标签对齐
  
- **`internal_metrics()`**: 内部指标（无需真实标签）
  - CH Score (Calinski-Harabasz)
  - DB Index (Davies-Bouldin)

### app/util.py - 工具函数

提供基础计算工具：

- **`l2_distance()`**: L2 距离计算（向量化实现）
- **`sample()`**: 随机采样
- **`get_labels()`**: 分配样本到最近中心
- **`cost()`**: 计算聚类损失
- **`k_nearest_neighbors()`**: K 近邻查找

### app/text.py - 文本处理工具

文本嵌入生成：

- **`get_text_cluster_data()`**: 加载和预处理文本数据
- **`output_embedding()`**: 使用 Sentence Transformer 生成嵌入向量

### config.py - 配置文件

定义全局路径和参数：

```python
SETTING = {
    'SEED': 1314,           # 随机种子
    'ROOT': 项目根目录，
    'RESULT': 结果输出目录，
    'DATA': 原始数据目录，
    'PROCESS_DATA': 预处理数据目录
}
```

---

## 技术栈

- **Python 3.8+**: 主要编程语言
- **PyTorch**: 深度学习框架（支持 CUDA 加速）
- **NumPy**: 高性能数值计算
- **Pandas**: 数据处理与 Excel 导出
- **scikit-learn**: 机器学习工具与评估指标
- **Sentence Transformers**: 文本嵌入模型
- **ModelScope**: 模型下载与管理
- **OpenPyXL**: Excel 文件读写

---

## 性能优化建议

### 1. GPU 加速

确保使用 GPU 加速可以显著提升性能：

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### 2. 批处理大小调整

根据内存大小调整批处理参数：

- **小内存 (<8GB)**: batch=256, total_batch=10
- **中内存 (8-16GB)**: batch=512, total_batch=20
- **大内存 (>16GB)**: batch=1024, total_batch=40

### 3. 并行化建议

虽然当前实现是单进程的，但可以通过以下方式提升效率：

- 减少迭代次数（`-i` 参数）
- 使用更小的 `rounds` 和 `trans` 值
- 在多个进程中并行运行不同的随机种子

---

## 注意事项

1. **预训练模型**：首次运行前确保下载好预训练模型
   ```bash
   modelscope download --model sentence-transformers/all-MiniLM-L6-v2
   ```

2. **数据格式**：数据集需要预先处理为 `.npy` 格式，可使用 `app/text.py` 生成

3. **GPU 加速**：建议使用 GPU 加速计算（支持 CUDA），CPU 模式会较慢

4. **内存管理**：大规模数据集运行时建议减小小批处理大小或增加总批次数

5. **结果去重**：结果文件会自动追加写入，相同参数的结果会覆盖旧数据

6. **随机种子**：所有随机操作使用固定种子（SEED=1314），保证结果可复现

---

## 常见问题 (FAQ)

### Q1: 如何选择合适的聚类数量？

A: 聚类数量会自动根据数据集的标签数确定。如果是自定义数据集，可以通过以下方式调整：

```python
# 在 app/run.py 中修改 k 的计算方式
k = len(labels)  # 自动根据标签数确定
# 或者手动指定
k = 10  # 固定为 10 个聚类
```

### Q2: 运行时内存不足怎么办？

A: 可以尝试以下方法：
- 减小 `batch` 参数：`-b 256`
- 减小 `trans` 参数：`-t 32`
- 减小 `rounds` 参数：`-r 200`
- 使用更小维度的嵌入模型

### Q3: 如何提高聚类质量？

A: 调优建议：
- 增加迭代次数：`-i 50` 或更高
- 增加 `rounds` 和 `trans` 参数
- 尝试不同的嵌入模型
- 比较归一化和未归一化的效果

### Q4: 如何使用自己的文本数据？

A: 步骤如下：

1. 准备数据：创建 `data/my_dataset/train.xlsx`，包含 'text' 和 'label' 列
2. 生成嵌入：
   ```bash
   python app/text.py -m all_minilm_l6 -d my_dataset
   ```
3. 运行聚类：
   ```bash
   python app/run.py -d my_dataset
   ```

### Q5: 结果中的各项指标含义是什么？

A: 各指标说明：

| 指标 | 全称 | 含义 | 取值范围 | 优劣 |
|------|------|------|----------|------|
| ARI | Adjusted Rand Index | 调整兰德指数 | [-1, 1] | 越大越好 |
| NMI | Normalized Mutual Information | 归一化互信息 | [0, 1] | 越大越好 |
| ACC | Accuracy | 准确率 | [0, 1] | 越大越好 |
| F1S | F1-Score | 调和平均数 | [0, 1] | 越大越好 |
| RS | Recall Score | 召回率 | [0, 1] | 越大越好 |
| PS | Precision Score | 精确率 | [0, 1] | 越大越好 |
| CH | Calinski-Harabasz | 簇间/簇内离散度比 | [0, +∞) | 越大越好 |
| DB | Davies-Bouldin | 簇内/簇间距离比 | [0, +∞) | 越小越好 |
| cost | Clustering Cost | 聚类损失函数 | [0, +∞) | 越小越好 |
| time | Running Time | 运行时间（秒） | [0, +∞) | 越短越好 |

### Q6: 为什么归一化和未归一化的结果不同？

A: 归一化会影响向量的分布特性：
- **归一化 (normalized)**: 向量长度为 1，适合余弦相似度
- **未归一化 (unnormlized)**: 保留原始向量长度信息

建议两种都尝试，选择效果更好的一种。

### Q7: 如何可视化聚类结果？

A: 可以使用 t-SNE 或 UMAP 进行降维可视化：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 加载数据
data = np.load('processed_data/ag_news/all_minilm_l6/norm_embedding.npy')
y = np.load('processed_data/ag_news/all_minilm_l6/y.npy')

# t-SNE 降维
tsne = TSNE(n_components=2, random_state=1314)
data_2d = tsne.fit_transform(data[:1000])  # 只可视化前 1000 个点

# 绘图
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=y[:1000], cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.savefig('visualization.png', dpi=300)
plt.show()
```

---

## 许可证

本项目仅供学术研究使用。

---

## 联系方式

如有问题，请通过邮件或 GitHub Issues 联系。
