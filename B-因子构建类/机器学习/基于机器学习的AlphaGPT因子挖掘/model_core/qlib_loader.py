""" 
 File: qlib_loader.py
 Date: 2026-01-17
 Description: Qlib 数据加载器。负责初始化 Qlib 环境，加载 A 股历史数据（开高低收量等），并将其转换为模型可用的 Tensor 格式。
 """ 
import torch
import qlib
from qlib.data import D
from qlib.constant import REG_CN
from .config import ModelConfig
from .factors import FeatureEngineer
import pandas as pd

class QlibDataLoader:
    def __init__(self, provider_uri='/Users/shuyan/.qlib/qlib_data/cn_data'):
        # 初始化 Qlib
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        self.raw_data_cache = None
        self.target_ret = None
        self.feat_tensor = None
        
    def load_data(self, start_time='2021-01-01', end_time='2023-12-31', instruments='csi300'):
        print(f"Loading A-Share data from Qlib ({start_time} to {end_time})...")
        
        # 1. 定义需要的字段
        # $factor 是复权因子，用于计算真实价格
        fields = ['$open', '$high', '$low', '$close', '$volume', '$factor']
        
        # 2. 获取股票池
        if isinstance(instruments, str):
            instruments = D.instruments(market=instruments)
        
        # 3. 加载数据
        # 返回 DataFrame: index=[datetime, instrument], columns=fields
        df = D.features(instruments, fields, start_time=start_time, end_time=end_time)
        
        if df.empty:
            raise ValueError("No data found from Qlib. Please check your data path and time range.")

        # 4. 数据转换: DataFrame -> Tensor (Assets, Time)
        # AlphaGPT 的 FeatureEngineer 期望输入形状为 (Assets, Time) 的 Tensor
        
        def to_tensor(field_name):
            # Unstack: index=datetime, columns=instrument
            # .T: index=instrument, columns=datetime -> (Assets, Time)
            # Log reveals D.features returns (instrument, datetime)
            # unstack(level=1) moves datetime to columns -> (instrument, datetime) i.e. (Assets, Time)
            # So we do NOT need .T if input is (instrument, datetime)
            # But to be safe, we check index names
            
            sub_df = df[field_name]
            if sub_df.index.names == ['instrument', 'datetime']:
                data = sub_df.unstack(level='datetime') # (Instrument, Datetime) -> (Assets, Time)
            else:
                # Assume (datetime, instrument)
                data = sub_df.unstack(level='instrument').T # (Datetime, Instrument) -> (Assets, Time)
            
            # 填充缺失值
            # 必须先用 ffill 再用 0 填充，确保没有 NaN
            data = data.ffill().fillna(0.0)
            
            # 检查是否仍有 NaN
            if data.isna().any().any():
                print(f"Warning: NaN detected in {field_name}, replacing with 0.0")
                data = data.fillna(0.0)
            
            return torch.tensor(data.values, dtype=torch.float32, device=ModelConfig.DEVICE)

        # 5. 构建 raw_data_cache，使用标准 key
        self.raw_data_cache = {
            'open': to_tensor('$open'),
            'high': to_tensor('$high'),
            'low': to_tensor('$low'),
            'close': to_tensor('$close'),
            'volume': to_tensor('$volume'),
            'factor': to_tensor('$factor'),
            # Qlib 数据通常不包含流动性/FDV，这里用 0 填充或用成交额替代
            'liquidity': to_tensor('$volume') * to_tensor('$close'), # 估算成交额
            'fdv': torch.zeros_like(to_tensor('$close'))
        }
        
        # 6. 计算特征
        print("Computing features...")
        # 注意：FeatureEngineer 可能需要修改以适应 (Assets, Time) 还是 (Time, Assets)？
        # 让我们检查一下 model_core/data_loader.py
        # 原版 loader: pivot = df.pivot(index='time', columns='address', values=col) => (Time, Assets)
        # 原版 return: torch.tensor(pivot.values.T) => (Assets, Time)
        # 所以我们的 to_tensor 返回 (Assets, Time) 是正确的。
        
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        
        # 7. 计算目标收益率 (T+1)
        # A股 T+1: 今天(T)买入，明天(T+1)才能卖出
        # 收益 = (Close_{T+2} * Adj_{T+2}) / (Close_{T+1} * Adj_{T+1}) - 1
        # 即预测明天的持仓在后天的收益
        
        cl = self.raw_data_cache['close']
        adj = self.raw_data_cache['factor']
        adj_close = cl * adj
        
        # T+1 Close
        t1 = torch.roll(adj_close, -1, dims=1) 
        # T+2 Close
        t2 = torch.roll(adj_close, -2, dims=1)
        
        # Log Return
        # 处理除零和 NaN 问题
        ret = torch.log(t2 / (t1 + 1e-9))
        self.target_ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        self.target_ret[:, -2:] = 0.0 # 最后两天无效
        
        print(f"Data Ready. Feature Shape: {self.feat_tensor.shape}")
