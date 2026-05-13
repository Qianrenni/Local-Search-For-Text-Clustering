## 目录
- [项目概述](#项目概述)
- [环境配置](#环境配置)
- [数据集准备](#数据集准备)
- [模型下载](#模型下载)
- [数据预处理](#数据预处理)
- [运行聚类算法](#运行聚类算法)
- [评估指标](#评估指标)
- [结果分析](#结果分析)
- [完整复现流程示例](#完整复现流程示例)
- [常见问题](#常见问题)
- [联系方式](#联系方式)

# 文本聚类算法研究项目

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

本项目实现了多种文本聚类算法，包括传统的KMeans、MiniBatchKMeans以及论文提出的Local Search算法。项目支持多种预训练语言模型进行文本嵌入，并在多个标准数据集上进行了实验验证。

## 项目概述

### 主要特性
- **多算法支持**: KMeans, MiniBatchKMeans, Local Search (论文核心算法)
- **多模型支持**: 支持多种Sentence Transformer模型进行文本嵌入
- **多数据集**: 支持20+个中英文数据集
- **全面评估**: 提供外部指标(ARI, NMI, ACC等)和内部指标(CH, DB, Cost等)
- **可扩展性**: 易于添加新数据集和新算法
- **可复现性**: 固定随机种子，确保实验结果可复现

### 项目结构
```
code/
├── app/                    # 核心算法实现
│   ├── kmeans.py          # KMeans算法
│   ├── mini_batch_kmeans.py # MiniBatchKMeans算法
│   ├── local_search.py    # Local Search算法核心
│   ├── run.py             # Local Search运行脚本
│   ├── text.py            # 文本嵌入生成
│   └── eval.py            # 评估指标计算
├── data/                   # 原始数据目录
├── preprocess_data/        # 预处理后数据
├── processed_data/         # 嵌入向量存储
├── result/                 # 实验结果
├── dependency/             # 模型依赖
└── config.py              # 配置文件
```

## 环境配置

### 依赖安装
```bash
pip install -r requirements.txt
```

主要依赖包括：
- `sentence-transformers`: 用于文本嵌入
- `scikit-learn`: 提供KMeans和MiniBatchKMeans算法及评估指标
- `numpy`, `pandas`: 数据处理
- `torch`: 深度学习框架
- `tqdm`: 进度条显示
- `openpyxl`: Excel文件读写
- `modelscope`: 模型库支持（可选）

### 系统要求
- Python 3.7+
- CUDA 11.8+ (可选，用于GPU加速)
- 至少8GB RAM (推荐16GB以上)

## 数据集准备

### 支持的数据集
项目支持以下数据集：
- `ag_news`: AG新闻分类数据集
- `20_newsgroups`: 20个新闻组数据集
- `bbc-news`: BBC新闻数据集
- `dbpedia_14`: DBpedia本体分类数据集
- `amazon_polarity`: 亚马逊情感极性数据集
- `imdb`: IMDB电影评论数据集
- `sst`: Stanford Sentiment Treebank
- `emotion`: 情感分类数据集
- `bank77`: 银行意图分类数据集
- `clue-tnews`: 中文新闻分类数据集
- `THUCNewsText`: 清华新闻文本数据集
- `amazon_massive`: Amazon MASSIVE多语言数据集
- `mtop_domain`: MTOP领域分类数据集
- `mtop_intent`: MTOP意图分类数据集
- 以及其他多个多语言和特定领域数据集

### 数据集下载方法

#### 方法一：使用Hugging Face Datasets（推荐）

大部分数据集可以通过Hugging Face Datasets库直接下载：

```bash
pip install datasets

# 设置镜像加速（国内用户推荐）
$env:HF_ENDPOINT = "https://hf-mirror.com"  # Windows PowerShell
# export HF_ENDPOINT="https://hf-mirror.com"  # Linux/Mac

# Python脚本下载示例
python -c "
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# 以ag_news为例
dataset = load_dataset('ag_news')
train_df = pd.DataFrame({
    'text': dataset['train']['text'],
    'label': dataset['train']['label']
})
train_df.to_excel('preprocess_data/ag_news/train.xlsx', index=False)
"
```

#### 方法二：使用Hugging Face CLI下载

```bash
pip install -U huggingface_hub

# 设置镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"  # Windows

# 下载数据集到本地
hf download ag_news --repo-type dataset --local-dir data/ag_news
```

#### 方法三：手动放置数据

对于某些特殊数据集（如THUCNewsText），需要手动处理：

1. 将原始数据放入 `data/{dataset_name}/` 目录
2. 使用提供的转换脚本处理数据格式
3. 生成 `preprocess_data/{dataset_name}/train.xlsx` 或 `train.json`

**示例：THUCNewsText数据处理**
```bash
# 确保data/THUCNewsText/目录下有parquet文件
python data/THUCNewsText/script.py
```

### 数据集格式要求

预处理后的数据需要满足以下格式：
- **Excel格式**: `preprocess_data/{dataset_name}/train.xlsx`
  - 必须包含 `text` 列（文本内容）
  - 必须包含 `label` 列（标签，可以是字符串或整数）
  
- **JSON格式**: `preprocess_data/{dataset_name}/train.json`
  - JSON数组，每个元素包含 `text` 和 `label` 字段

### 数据集位置结构

```
data/                          # 原始数据目录
├── ag_news/
├── 20_newsgroups/
└── ...

preprocess_data/               # 预处理后数据目录
├── ag_news/
│   └── train.xlsx            # 训练数据（必需）
├── 20_newsgroups/
│   └── train.xlsx
└── ...

processed_data/                # 嵌入向量目录（自动生成）
├── ag_news/
│   ├── all-MiniLM-L6-v2/
│   │   ├── norm_embedding.npy
│   │   ├── unnormlized_embedding.npy
│   │   ├── y.npy
│   │   └── labels.json
│   └── ...
└── ...
```

## 模型下载

### 支持的嵌入模型
项目支持以下Sentence Transformer模型：
- `all-MiniLM-L6-v2`: 轻量级英文模型（默认）
- `all-MiniLM-L12-v2`: 中等规模英文模型
- `all-mpnet-base-v2`: 高性能英文模型
- `paraphrase-multilingual-MiniLM-L12-v2`: 多语言模型
- `gte-base`: 通用文本嵌入模型
- `gte-base-zh`: 中文专用模型

### 模型下载方法
模型会自动从Hugging Face Hub下载到 `dependency/` 目录。如需手动下载：

```bash
pip install -U huggingface_hub
# 设置镜像（国内用户推荐）
$env:HF_ENDPOINT = "https://hf-mirror.com"  # Windows PowerShell
# export HF_ENDPOINT="https://hf-mirror.com"  # Linux/Mac

# 下载模型
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir dependency/all-MiniLM-L6-v2
```

## 数据预处理

### 生成文本嵌入
使用 `app/text.py` 脚本为数据集生成嵌入向量：

```bash
# 基本用法：为ag_news数据集使用默认模型生成嵌入
python app/text.py -d ag_news

# 指定模型和数据集
python app/text.py -d ag_news -m all-MiniLM-L6-v2

# 使用归一化嵌入
python app/text.py -d ag_news -m all-MiniLM-L6-v2 -n 1

# 为所有可用模型生成嵌入
python app/text.py -d ag_news -a 1

# 自定义批处理大小
python app/text.py -d ag_news -b 128
```

### 参数说明
- `-d, --dataset_name`: 数据集名称（默认: ag_news）
- `-m, --model`: 模型名称（默认: all-MiniLM-L6-v2）
- `-b, --batch_size`: 批处理大小（默认: 64）
- `-n, --norm`: 是否归一化嵌入（0: 否, 1: 是，默认: 0）
- `-a, --all`: 是否为所有模型生成嵌入（0: 否, 1: 是，默认: 0）
- `-t, --truncated`: 是否截断长文本（0: 使用均值池化, 1: 截断，默认: 1）

### 输出文件
预处理后会在 `processed_data/{dataset_name}/{model_name}/` 目录下生成：
- `norm_embedding.npy` 或 `unnormlized_embedding.npy`: 嵌入向量
- `y.npy`: 真实标签
- `labels.json`: 标签映射

## 运行聚类算法

项目实现了三种聚类算法，分别通过不同的脚本运行：

### 1. KMeans 算法
```bash
# 基本用法
python app/kmeans.py -d ag_news

# 指定模型和簇数
python app/kmeans.py -d ag_news -m all-MiniLM-L6-v2 -k 4

# 自定义迭代次数和容忍度
python app/kmeans.py -d ag_news -r 100 -t 0.0001

# 运行多次实验取平均
python app/kmeans.py -d ag_news -i 30

# 对所有模型运行
python app/kmeans.py -d ag_news -a 1

# 使用归一化嵌入
python app/kmeans.py -d ag_news -n 1
```

**参数说明：**
- `-d, --dataset`: 数据集名称
- `-m, --model`: 模型名称
- `-k, --clusters`: 簇数量（-1表示使用真实标签数）
- `-i, --iteration`: 实验重复次数（默认: 30）
- `-r, --rounds`: 最大迭代轮数
- `-t, --tol`: 收敛容忍度（默认: 0.001）
- `-n, --norm`: 使用归一化嵌入（0: 否, 1: 是）
- `-a, --all`: 对所有模型运行

### 2. MiniBatchKMeans 算法
```bash
# 基本用法
python app/mini_batch_kmeans.py -d ag_news

# 指定批次大小
python app/mini_batch_kmeans.py -d ag_news -b 256

# 自定义参数
python app/mini_batch_kmeans.py -d ag_news -k 4 -r 50 -t 0.0001 -b 512
```

**参数说明：**
- `-b, --batch`: 小批量大小
- 其他参数与KMeans相同

### 3. Local Search 算法（论文核心算法）

Local Search算法是本项目的核心创新点，通过局部搜索策略优化聚类中心，通常能获得比传统KMeans更好的聚类质量。

**算法原理：**
- 使用MiniBatch KMeans进行初始化
- 通过局部搜索策略迭代优化聚类中心
- 引入bandit机制平衡探索与利用
- 支持自适应批次大小和变换次数

```bash
# 基本用法
python app/run.py -d ag_news

# 自定义算法参数
python app/run.py -d ag_news -r 60 -t 10 -b 1024 -tb 20

# 指定Minibatch轮数
python app/run.py -d ag_news -mbr 30

# 完整参数示例
python app/run.py -d ag_news \
    -m all-MiniLM-L6-v2 \
    -k 4 \
    -i 30 \
    -r 60 \
    -t 10 \
    -b 1024 \
    -tb 20 \
    -mbr 30 \
    -n 1
```

**参数说明：**
- `-r, --rounds`: 搜索轮数（默认: min(60, k*15)）
- `-t, --trans`: 变换次数（默认: ceil(sqrt(rounds)) + k）
- `-b, --batch`: 批次大小（默认: min(1024, 128*k)）
- `-tb, --total_batch`: 总批次大小（默认: 15）
- `-mbr, --minibatch_rounds`: Minibatch轮数（默认: rounds//2）
- `-th, --threshold`: 阈值参数（默认: 0.005）
- `-n, --norm`: 使用归一化嵌入

### 4. Terminate MiniBatch KMeans 算法
```bash
# 基本用法
python app/run_terminate_mini_batch.py -d ag_news

# 自定义参数
python app/run_terminate_mini_batch.py -d ag_news -r 50 -b 256
```

**参数说明：**
- `-r, --rounds`: 迭代轮数
- `-b, --batch`: 批次大小
- 其他参数与MiniBatchKMeans类似

### 算法选择建议
- **KMeans**: 基准算法，适合中小规模数据集
- **MiniBatchKMeans**: 适合大规模数据集，速度更快
- **Local Search**: 论文提出的改进算法，通常能获得更好的聚类质量
- **Terminate MiniBatch KMeans**: 带终止条件的MiniBatch KMeans变体

## 评估指标

### 外部指标（需要真实标签）
1. **ARI (Adjusted Rand Index)**: 调整兰德指数
   - 范围: [-1, 1]，越接近1越好
   - 衡量聚类结果与真实标签的一致性

2. **NMI (Normalized Mutual Information)**: 归一化互信息
   - 范围: [0, 1]，越接近1越好
   - 衡量聚类结果与真实标签的信息共享程度

3. **ACC (Accuracy)**: 聚类准确率
   - 范围: [0, 1]，越接近1越好
   - 使用匈牙利算法找到最优标签映射后计算

4. **F1S (F1-Score)**: 加权F1分数
   - 范围: [0, 1]，越接近1越好

5. **RS (Recall Score)**: 加权召回率
6. **PS (Precision Score)**: 加权精确率

### 内部指标（不需要真实标签）
1. **CH (Calinski-Harabasz Index)**: CH指数
   - 范围: [0, +∞)，越大越好
   - 衡量簇间离散度与簇内离散度的比值

2. **DB (Davies-Bouldin Index)**: DB指数
   - 范围: [0, +∞)，越小越好
   - 衡量簇内距离与簇间距离的比值

3. **Cost**: 聚类损失函数值
   - 越小越好，表示样本到簇中心的距离之和

4. **Time**: 运行时间（秒）
   - 衡量算法效率

## 结果分析

### 结果存储
实验结果保存在 `result/{dataset_name}/{algorithm}/` 目录下：
- `data.xlsx`: 每次实验的详细结果
- `aggregate_data.xlsx`: 聚合统计结果（均值和标准差）

### 结果文件格式
每个Excel文件包含以下列：
- `dataset`: 数据集名称
- `datetime`: 实验时间
- `model`: 使用的嵌入模型
- `norm`: 是否使用归一化嵌入
- `clusters`: 簇数量
- `rounds/trans/batch/tol`: 算法超参数
- `iteration`: 实验迭代编号
- `ARI/NMI/DB/CH/ACC/F1S/RS/PS`: 评估指标
- `cost`: 聚类损失
- `time`: 运行时间

### 结果可视化
您可以使用以下工具对结果进行分析：
- Excel内置图表功能
- Python的matplotlib/seaborn库
- Jupyter Notebook进行交互式分析

## 完整复现流程示例

以AG News数据集为例，完整复现流程如下：

```bash
# 1. 安装依赖
pip install -r requirements.txt
pip install datasets  # 用于下载数据集

# 2. 下载并预处理数据集
# 方法一：使用Hugging Face自动下载
python -c "
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# 设置镜像（如果需要）
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载ag_news数据集
dataset = load_dataset('ag_news')
train_df = pd.DataFrame({
    'text': dataset['train']['text'],
    'label': dataset['train']['label']
})

# 创建目录并保存
Path('preprocess_data/ag_news').mkdir(parents=True, exist_ok=True)
train_df.to_excel('preprocess_data/ag_news/train.xlsx', index=False)
print('数据集下载完成！')
"

# 方法二：如果已有train.xlsx文件，跳过此步骤

# 3. 下载模型（自动）
# 模型会在首次运行时自动下载到 dependency/ 目录
# 也可手动下载：
# hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir dependency/all-MiniLM-L6-v2

# 4. 生成嵌入向量
python app/text.py -d ag_news -m all-MiniLM-L6-v2 -n 0
python app/text.py -d ag_news -m all-MiniLM-L6-v2 -n 1

# 5. 运行三种算法
# KMeans
python app/kmeans.py -d ag_news -m all-MiniLM-L6-v2 -i 30

# MiniBatchKMeans
python app/mini_batch_kmeans.py -d ag_news -m all-MiniLM-L6-v2 -i 30

# Local Search
python app/run.py -d ag_news -m all-MiniLM-L6-v2 -i 30

# 6. 查看结果
# 结果保存在 result/ag_news/{kmeans|mini_batch_kmeans|local_search}/ 目录
```

## 常见问题

### Q1: 如何添加新数据集？
1. 将原始数据放入 `data/{dataset_name}/` 目录
2. 创建预处理脚本，生成 `preprocess_data/{dataset_name}/train.xlsx`
3. 确保Excel文件包含 `text` 和 `label` 两列
4. 运行 `python app/text.py -d {dataset_name}` 生成嵌入向量

### Q2: GPU加速
代码会自动检测并使用GPU（如果可用）。确保已安装CUDA版本的PyTorch：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Q3: 内存不足
- 减小批处理大小 (`-b` 参数)
- 使用更小的嵌入模型（如 `all-MiniLM-L6-v2`）
- 对大数据集使用MiniBatchKMeans
- 考虑使用均值池化方式处理长文本 (`-t 0`)

### Q4: 复现性
所有随机种子已固定为 `1314`（在 `config.py` 中定义），确保实验可复现。

### Q5: 如何处理中文数据集？
1. 使用支持中文的模型，如 `gte-base-zh`
2. 确保数据集编码为UTF-8
3. 参考 `clue-tnews` 或 `THUCNewsText` 的处理方式

### Q6: 如何比较不同算法的效果？
1. 在同一数据集上运行所有算法
2. 查看 `result/{dataset_name}/{algorithm}/aggregate_data.xlsx`
3. 比较ARI、NMI、ACC等外部指标
4. 比较CH、DB、Cost等内部指标
5. 对比运行时间

本项目仅供学术研究使用。

## 引用

如果您在研究中使用了本项目，请引用相关论文：

```bibtex
@inproceedings{paper,
  title={Your Paper Title},
  author={Author Name},
  booktitle={Conference Name},
  year={2026}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 致谢

感谢以下开源项目的支持：
- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Scikit-learn](https://scikit-learn.org/)

## 联系方式

如有问题，请提交Issue或联系项目维护者。

## 贡献指南

欢迎为本项目做出贡献！如果您想参与开发：

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

### 开发建议
- 保持代码风格一致
- 添加必要的注释和文档
- 编写测试用例
- 更新README文档