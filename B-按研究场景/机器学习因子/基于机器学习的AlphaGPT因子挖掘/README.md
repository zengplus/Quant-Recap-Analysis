# AlphaGPT-AShare

> [!NOTE]
>  **基于 [AlphaGPT](https://github.com/imbue-bit/AlphaGPT)项目的基础上学习**

---
## 项目简介

基于 Transformer 与强化学习的智能因子挖掘系统，融合符号回归与强化学习奖励机制，自动挖掘超额收益量化因子。

## 聚宽使用

- 请务必先将训练好的模型文件 (例如 alphagpt_model.npz) 上传到聚宽(JoinQuant)的研究环境根目录下。

- 然后将（joinquant_strategy.py）的所有代码复制粘贴到聚宽的策略回测中。

- 编译运行回测即可。

## 架构解析

### 1. 输入特征工程体系

基于原版设计的基础特征，这些特征覆盖了价格、流动性、情绪等多个维度：

| 特征 | 计算公式 | 市场含义 | 工程化处理 |
|------|----------|----------|------------|
| **收益率** | `log(Close_t / Close_{t-1})` | 价格趋势动量 | 对数变换，平滑极端值 |
| **流动性健康度** | `Volume / MarketCap` | 资金活跃程度 | 比率标准化，去除规模效应 |
| **买卖压力** | `(Close - Open) / (High - Low)` | 多空力量对比 | 避免除零错误，[-1,1]归一化 |
| **情绪加速** | `∇²Volume` | 市场情绪变化率 | 二阶差分，捕捉拐点 |
| **偏离度** | `(Close - MA20) / σ20` | 超买超卖状态 | Z-score标准化 |
| **对数成交量** | `log(Volume + 1)` | 交易活跃度 | 对数变换，降低偏度 |

### 2. 核心算法：Transformer + 强化学习

#### 2.1 数据流管道

原始数据 → 特征工程 → Transformer生成器 → 因子表达式 → 回测引擎 → 奖励信号 → 策略梯度更新

#### 2.2 Transformer 作为序列生成器
- **输入序列**：`[START, Feature1, Operator, Feature2, ...]`
- **输出序列**：数学表达式符号流
- **词汇表设计**：
  - 算子：`+`, `-`, `×`, `÷`, `Mean(,5)`, `Std(,10)`, `Rank()`, `Delay(,1)`
  - 函数：`Abs()`, `Log()`, `Sign()`, `Max(,)`, `Min(,)`
  - 特征：`Returns`, `Liquidity`, `Pressure`, `FOMO`, `Deviation`, `LogVolume`

#### 2.2 强化学习奖励机制
系统采用**近端策略优化（PPO）**算法，奖励函数设计为多目标优化：


Reward = w₁×Sharpe + w₂×Sortino + w₃×IC - w₄×Turnover - w₅×Complexity


权重配置：
- `w₁ = 0.4`：夏普比率（风险调整后收益）
- `w₂ = 0.3`：索提诺比率（下行风险调整）
- `w₃ = 0.2`：信息系数（预测能力）
- `w₄ = 0.05`：换手率惩罚（控制交易成本）
- `w₅ = 0.05`：复杂度惩罚（防止过拟合）

#### 2.3 Low-Rank Decay (LoRD) 正则化
为防止过拟合，我们引入了创新的 LoRD 技术：

```python
# 传统 L2 正则化：惩罚所有权重
loss = cross_entropy + λ∑θ²

# LoRD 正则化：选择性惩罚低频组合
loss = cross_entropy + λ∑(UΣVᵀ)²  # 对权重矩阵的低秩近似施加惩罚
```

这种方法特别适合因子挖掘任务，因为它：
1. 保留高频、有效的算子组合
2. 抑制随机、偶然的噪声组合
3. 提升模型在样本外的泛化能力


## 快速开始指南

### 第一步：环境配置

#### 方案 A：使用 uv
```bash
# 1. 安装 uv（一次性操作）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 进入项目目录并同步环境
uv sync --locked

# 3. 激活环境
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows
```

#### 方案 B：使用 pip
```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活环境
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

### 第二步：数据准备

本项目依赖 Qlib 数据格式的 A 股历史数据。

> **注意**：推荐**手动下载并配置数据**。

1.  **手动下载数据**：
    访问 [chenditc/investment_data Releases](https://github.com/chenditc/investment_data/releases) 页面。
    下载最新的 `qlib_bin.tar.gz` (或其他类似的 Qlib 数据压缩包)。

2.  **解压数据**：
    将下载的压缩包解压的目录（默认路径：`~/.qlib/qlib_data/cn_data`）。
    
    *注意：`--strip-components=1` 用于去除压缩包内的一层父目录（如果有），请根据实际解压情况调整，确保 `cn_data` 目录下直接包含 `calendars`, `features`, `instruments` 等文件夹。*

3.  **配置路径**：
    本项目默认会在 `~/.qlib/qlib_data/cn_data` 寻找数据。
    如果你将数据解压到了其他位置，请修改 `model_core/qlib_loader.py` 文件中的 `provider_uri` 参数：
    
    ```python
    # model_core/qlib_loader.py
    class QlibDataLoader:
        def __init__(self, provider_uri='/你的/自定义/数据路径'):
            # ...
    ```

### 第三步：首次训练（30-60分钟）

```bash
# 基础训练（使用默认参数）
python train_ashare.py

# 进阶：指定训练周期
python train_ashare.py \
    --train_start 2020-01-01 \
    --train_end 2022-12-31 \
    --valid_start 2023-01-01 \
    --valid_end 2023-06-30
```

### 第四步：回测验证（5分钟）

```bash
python run_qlib_backtest.py
```

## 配置详解

### 训练配置（train_ashare.py）
```python
class ModelConfig:
    # 数据配置
    UNIVERSE = "csi300"  # 标的池：csi300, csi500, all
    FREQUENCY = "daily"  # 数据频率：daily, 60min, 30min
    
    # 时间范围
    TRAIN_START = "2020-01-01"
    TRAIN_END = "2022-12-31"
    VALID_START = "2023-01-01"
    VALID_END = "2023-06-30"
    
    # 模型参数
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPISODES = 10000  # 训练回合数
    HIDDEN_SIZE = 256  # Transformer隐层维度
    N_LAYERS = 6  # Transformer层数
    N_HEADS = 8  # 注意力头数
    
    # 强化学习参数
    REWARD_WEIGHTS = [0.4, 0.3, 0.2, 0.05, 0.05]  # 奖励权重
    GAMMA = 0.99  # 折扣因子
    CLIP_EPSILON = 0.2  # PPO裁剪参数
```

### 回测配置（run_qlib_backtest.py）
```python
class BacktestConfig:
    # 回测周期
    START_DATE = "2023-01-01"
    END_DATE = "2023-12-31"
    
    # 资金配置
    INITIAL_CAPITAL = 1000000  # 初始资金100万
    COMMISSION_RATE = 0.0003  # 佣金万分之三
    STAMP_TAX_RATE = 0.001  # 印花税千分之一
    
    # 风控参数
    MAX_POSITION_RATIO = 0.1  # 单票最大仓位10%
    STOP_LOSS = 0.08  # 止损线8%
    STOP_PROFIT = 0.25  # 止盈线25%
    
    # 交易约束
    MIN_TRADE_AMOUNT = 10000  # 最小交易金额1万
    REQUIRE_T_PLUS_1 = True  # T+1约束
    HANDLE_LIMIT_UP_DOWN = True  # 处理涨跌停
```

## 因子解读示例

系统挖掘出的因子通常具有可解释性。例如：

```json
{
  "factor_name": "AlphaGPT_v1_023",
  "expression": "Rank(Mean(LogVolume, 5)) - Delay(Pressure, 1) × 0.3",
  "interpretation": {
    "核心逻辑": "成交量排名与买卖压力的动态调节",
    "第一部分": "Rank(Mean(LogVolume, 5)) - 过去5日成交量的排名，识别持续放量股票",
    "第二部分": "Delay(Pressure, 1) × 0.3 - 昨日买卖压力的30%，避免追涨杀跌",
    "市场含义": "寻找成交量持续放大但买卖压力不过热的股票",
    "预期表现": "在震荡市中捕捉量价背离机会"
  },
  "performance": {
    "in_sample_sharpe": 2.1,
    "out_of_sample_sharpe": 1.8,
    "ic_mean": 0.08,
    "ic_ir": 2.5
  }
}
```


---
*免责声明：本项目仅供学习和研究使用。任何基于本项目的投资决策，用户需自行承担风险。作者不对任何投资损失负责。*