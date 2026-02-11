""" 
 File: qlib_loader.py
 Date: 2026-01-17
 Description: Qlib 数据加载器。负责初始化 Qlib 环境，加载 A 股历史数据（开高低收量等），并将其转换为模型可用的 Tensor 格式。
 """ 
import os
import torch
import logging
import contextlib
import warnings
os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
from .config import ModelConfig
from .factors import FeatureEngineer
import pandas as pd
import numpy as np

_DEVNULL = None


def _devnull_stream():
    dn = globals().get("_DEVNULL")
    if dn is None or getattr(dn, "closed", False):
        dn = open(os.devnull, "w")
        globals()["_DEVNULL"] = dn
    return dn


def _filter_kcbj_instruments(instruments):
    try:
        items = list(instruments)
    except Exception:
        return instruments
    if not items:
        return instruments
    if not all(isinstance(x, str) for x in items):
        return instruments
    keep = []
    for s in items:
        core = s
        if len(core) >= 2 and core[:2] in ("SH", "SZ", "BJ"):
            core = core[2:]
        if core.startswith("688") or core.startswith("8") or core.startswith("4"):
            continue
        keep.append(s)
    return keep


class QlibDataLoader:
    def __init__(self, provider_uri='/Users/shuyan/.qlib/qlib_data/cn_data/qlib_bin'):
        logging.getLogger("qlib").setLevel(logging.ERROR)
        logging.getLogger("qlib").propagate = False
        logging.getLogger("gym").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="qlib.utils.index_data")
        dn = _devnull_stream()
        with contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            import qlib
            from qlib.data import D
            from qlib.constant import REG_CN
            from qlib.config import C
            self._qlib = qlib
            self._D = D
            self._REG_CN = REG_CN
            self._qlib.init(provider_uri=provider_uri, region=self._REG_CN)
            C.kernels = 1
            C.joblib_backend = "threading"
        self.raw_data_cache = None
        self.target_ret = None
        self.feat_tensor = None
        self.dates = None
        self.assets = None
        
    def load_data(self, start_time='2021-01-01', end_time='2023-12-31', instruments='csi300', verbose=True, price_mode="raw"):
        if verbose:
            print(f"Loading A-Share data from Qlib ({start_time} to {end_time})...")
        
        # 1. 定义需要的字段
        # $factor 是复权因子，用于计算真实价格
        fields = ['$open', '$high', '$low', '$close', '$volume', '$factor', '$amount']
        
        # 2. 获取股票池
        if isinstance(instruments, str):
            instruments = self._D.instruments(market=instruments)
        
        # Resolve to list using D.list_instruments
        # This handles both config dicts (market='csi300') and symbol dicts
        try:
            instruments = self._D.list_instruments(instruments=instruments, start_time=start_time, end_time=end_time, as_list=True)
        except Exception as e:
            print(f"D.list_instruments failed: {e}")
            # Fallback: if it's a dict, take keys
            if isinstance(instruments, dict):
                instruments = list(instruments.keys())
        
        # Debug info
        if hasattr(instruments, '__len__'):
             print(f"Raw instruments count: {len(instruments)}")
        else:
             print(f"Raw instruments type: {type(instruments)}")
             
        instruments = _filter_kcbj_instruments(instruments)
        print(f"Filtered instruments count: {len(instruments)}")
        
        if len(instruments) == 0:
             raise ValueError("No instruments left after filtering!")

        # 3. 加载数据
        # 返回 DataFrame: index=[datetime, instrument], columns=fields
        try:
            df = self._D.features(instruments, fields, start_time=start_time, end_time=end_time)
        except Exception as e:
            print(f"D.features failed: {e}")
            raise
            
        print(f"Loaded DataFrame shape: {df.shape}")
        
        if df.empty:
            print(f"Start: {start_time}, End: {end_time}")
            print(f"Instruments sample: {instruments[:5] if instruments else 'None'}")
            raise ValueError("No data found from Qlib. Please check your data path and time range.")

        # 保存日期索引 (Time)
        # 确保我们获取的是唯一的日期列表，且排序
        if 'datetime' in df.index.names:
             self.dates = df.index.get_level_values('datetime').unique().sort_values()
        else:
             # Fallback if index names are messed up, but usually D.features returns MultiIndex
             self.dates = pd.to_datetime(df.index.get_level_values(0).unique()).sort_values()

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
            data = data.ffill(axis=1).fillna(0.0)
            if self.assets is None:
                self.assets = list(data.index)
                self.stock_list = self.assets # Alias for compatibility
            
            # 检查是否仍有 NaN
            if data.isna().any().any() and verbose:
                print(f"Warning: NaN detected in {field_name}, replacing with 0.0")
                data = data.fillna(0.0)
            
            return torch.tensor(data.values, dtype=torch.float32, device=ModelConfig.DEVICE)

        open_t = to_tensor('$open')
        high_t = to_tensor('$high')
        low_t = to_tensor('$low')
        close_t = to_tensor('$close')
        volume_t = to_tensor('$volume')
        factor_t = to_tensor('$factor')
        amount_t = to_tensor('$amount')

        close_raw = close_t

        price_mode_norm = str(price_mode).lower().strip() if price_mode is not None else "raw"
        factor_safe = torch.where(factor_t > 1e-9, factor_t, torch.ones_like(factor_t))
        if price_mode_norm in {"pre_adjusted", "pre", "adjusted", "adj"}:
            open_t = open_t * factor_safe
            high_t = high_t * factor_safe
            low_t = low_t * factor_safe
            close_t = close_t * factor_safe
        elif price_mode_norm in {"raw", "real", "unadjusted"}:
            pass

        # 5. 构建 raw_data_cache，使用标准 key
        self.raw_data_cache = {
            'open': open_t,
            'high': high_t,
            'low': low_t,
            'close': close_t,
            'volume': volume_t,
            'factor': factor_safe,
            # Qlib 数据通常不包含流动性/FDV，这里用 0 填充或用成交额替代
            'liquidity': amount_t,
            'fdv': torch.zeros_like(close_t)
        }
        
        # 6. 计算特征
        if verbose:
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
        
        close_for_return = close_raw
        t1 = torch.roll(close_for_return, -1, dims=1)

        valid = (close_for_return > 1e-6) & (t1 > 1e-6)
        safe = torch.where(valid, t1 / (close_for_return + 1e-9), torch.ones_like(t1))
        ret = torch.where(valid, torch.log(safe), torch.zeros_like(t1))
        self.target_ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        self.target_ret[:, -1:] = 0.0 # 最后一天无效

        bench = getattr(ModelConfig, "BENCHMARK", "SH000300")
        bench_ret = None
        try:
            bench_df = self._D.features([bench], ['$close'], start_time=start_time, end_time=end_time)
            if bench_df.index.names == ['instrument', 'datetime']:
                bench_close = bench_df['$close'].unstack(level='datetime').iloc[0]
            else:
                bench_close = bench_df['$close'].unstack(level='instrument').iloc[:, 0]
            bench_close.index = pd.to_datetime(bench_close.index)
            bench_close = bench_close.reindex(pd.to_datetime(self.dates)).ffill().fillna(0.0)
            bench_ret_s = (bench_close.shift(-1) / (bench_close + 1e-9)).replace([float("inf"), -float("inf")], 1.0)
            bench_ret_s = np.log(bench_ret_s).fillna(0.0)
            bench_ret_s.iloc[-1] = 0.0
            bench_ret = torch.tensor(bench_ret_s.values, dtype=torch.float32, device=ModelConfig.DEVICE)
        except Exception:
            bench_ret = None
        if bench_ret is not None:
            self.raw_data_cache["bench_ret"] = bench_ret
        
        if verbose:
            print(f"Data Ready. Feature Shape: {self.feat_tensor.shape}")
