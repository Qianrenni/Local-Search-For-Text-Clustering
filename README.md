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

1. **初始化**：随机采样 k 个初始聚类中心
2. **局部搜索优化**：
   - 在每轮迭代中探索 `trans` 个候选点
   - 使用 Bandit 算法评估候选点质量
   - 选择最优候选点更新聚类中心
3. **Mini-batch K-means 微调**：对最终中心点进行精细调整
4. **结果评估**：计算多种聚类和评估指标
5. **输出生成**：将结果保存为 Excel 文件

### 核心参数

- `n_clusters`: 聚类数量（自动根据数据集标签数确定）
- `rounds`: 局部搜索迭代轮数（默认：k×100 或数据量×20%）
- `trans`: 候选点探索步数（默认：64）
- `batch`: 批处理大小（默认：512）
- `minibatch_rounds`: Mini-batch 微调轮数（默认：40）

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

## 自定义数据集

如需添加新数据集，请按以下格式准备数据：

1. 在 `processed_data/` 下创建数据集目录
2. 为每个嵌入模型创建子目录
3. 包含以下文件：
   - `unnormlized_embedding.npy` - 原始嵌入向量
   - `norm_embedding.npy` - 归一化嵌入向量
   - `y.npy` - 真实标签
   - `labels.json` - 标签名称列表

---

## 技术栈

- **Python**: 主要编程语言
- **PyTorch**: 深度学习框架（支持 CUDA 12.9）
- **NumPy**: 数值计算
- **Pandas**: 数据处理与结果导出
- **scikit-learn**: 机器学习工具与评估指标
- **ModelScope**: 模型下载与管理
- **Matplotlib**: 可视化（可选）

---

## 注意事项

1. 首次运行前确保下载好预训练模型
2. 数据集需要预先处理为 `.npy` 格式
3. 建议使用 GPU 加速计算（支持 CUDA）
4. 大规模数据集运行时建议减少迭代次数
5. 结果文件会自动去重并追加写入

---

## 许可证

本项目仅供学术研究使用。

---

## 联系方式

如有问题，请通过邮件或 GitHub Issues 联系。
