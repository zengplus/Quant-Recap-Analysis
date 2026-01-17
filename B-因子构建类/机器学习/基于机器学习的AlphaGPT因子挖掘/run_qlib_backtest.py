""" 
 File: run_qlib_backtest.py
 Date: 2026-01-17
 Description: Qlib 回测脚本。加载训练好的策略公式，使用 Qlib 框架进行全量回测，计算因子值并生成回测报告。
 """ 
import warnings
# Suppress Gym warnings (safe to ignore since we downgraded numpy to <2.0)
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

import torch
import json
import pandas as pd
import qlib

from qlib.data import D
from qlib.constant import REG_CN
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy

from model_core.config import ModelConfig
from model_core.qlib_loader import QlibDataLoader
from model_core.vm import StackVM

def run_backtest():
    # 1. 加载策略
    print("📂 Loading best strategy...")
    try:
        with open("best_ashare_strategy.json", "r") as f:
            formula = json.load(f)
        print(f"Strategy: {formula}")
    except FileNotFoundError:
        print("❌ best_ashare_strategy.json not found. Please run train_ashare.py first.")
        return

    # 2. 准备回测数据 (使用全量数据或测试集)
    # 建议使用与训练不同的时间段，或者包含训练集的全量回测
    start_time = '2022-01-01'
    end_time = '2023-12-31'
    
    # 复用 QlibDataLoader 以保证特征计算一致性
    loader = QlibDataLoader()
    loader.load_data(start_time=start_time, end_time=end_time, instruments='csi300')
    
    # 3. 计算因子值 (Signal)
    print("🧮 Computing factor scores...")
    vm = StackVM()
    # 执行公式得到因子值 (Assets, Time)
    factor_scores = vm.execute(formula, loader.feat_tensor)
    
    if factor_scores is None:
        print("❌ Failed to compute factor scores.")
        return

    # 4. 转换为 Qlib 格式 (DataFrame: Index=[datetime, instrument], Column='score')
    # 我们需要重构 DataFrame 结构
    
    # 获取原始 DataFrame 的索引结构
    instruments = D.instruments(market='csi300')
    fields = ['$close'] # 只需要索引，字段无所谓
    raw_df = D.features(instruments, fields, start_time=start_time, end_time=end_time)
    
    print("Raw DF Index:", raw_df.index)
    print("Raw DF Index Levels:", raw_df.index.names)

    if raw_df.index.names == ['instrument', 'datetime']:
        unstacked = raw_df['$close'].unstack(level='datetime')
    else:
        unstacked = raw_df['$close'].unstack(level='instrument').T

    print("Unstacked Index:", unstacked.index)
    print("Unstacked Columns:", unstacked.columns)

    # unstacked: index=Instrument (Assets), columns=Datetime (Time)
    asset_list = unstacked.index
    time_list = unstacked.columns

    # factor_scores is (Assets, Time) -> Transpose to (Time, Assets)
    scores_t = factor_scores.T.cpu().numpy() # (Time, Assets)
    
    # 构建 Score DataFrame
    # 确保索引是 DatetimeIndex
    time_list = pd.to_datetime(time_list)
    score_df = pd.DataFrame(scores_t, index=time_list, columns=asset_list)
    
    # Stack 回去变成 (datetime, instrument) 的 MultiIndex
    pred_df = score_df.stack().to_frame('score')
    pred_df.index.names = ['datetime', 'instrument']
    
    # 过滤掉无法交易的股票 (停牌/无数据)
    # raw_df 中 $close 为 NaN 的地方说明无法交易
    # 注意：raw_df 可能和 pred_df 索引顺序不完全一致，需要对齐
    
    # 确保 raw_df 也是 (datetime, instrument)
    if raw_df.index.names == ['instrument', 'datetime']:
        raw_df_aligned = raw_df.swaplevel().sort_index()
    else:
        raw_df_aligned = raw_df.sort_index()
        
    # 合并数据以进行过滤
    # 使用 inner join 确保只保留有行情数据的点
    merged_df = pred_df.join(raw_df_aligned['$close'], how='inner')
    
    # 进一步过滤掉 close 为 NaN 的点 (虽然 inner join 可能已经处理了一部分，但显式过滤更安全)
    valid_mask = ~merged_df['$close'].isna()
    pred_df = merged_df.loc[valid_mask, ['score']]
    
    # 确保排序
    pred_df = pred_df.sort_index()

    print(f"Signal Ready. Shape: {pred_df.shape}")
    print(pred_df.head())
    print("Index types:", pred_df.index.get_level_values(0).dtype, pred_df.index.get_level_values(1).dtype)


    # 5. 配置 Qlib 回测
    print("🚀 Starting Qlib Backtest...")
    
    # 策略配置: Top 30, 每日换仓
    strategy_config = {
        "topk": 30,
        "n_drop": 30, # 每日全换 (Aggressive) 或者设置小一点实现增量换仓
        "signal": pred_df,
    }
    
    # 账户配置
    account_config = {
        "account": 1000000,
        "benchmark": "SH000300", # 沪深300
    }
    
    # 运行回测
    # 使用 backtest 函数
    report_normal, positions_normal = backtest_daily(
        start_time=start_time, 
        end_time=end_time, 
        strategy=TopkDropoutStrategy(**strategy_config),
        account=1000000,
        benchmark='SH000300',
    )
    
    # 6. 分析结果
    print("\n📊 Backtest Results Analysis:")
    analysis = risk_analysis(report_normal['return'] - report_normal['bench'])
    
    print("--- Excess Return Analysis (Alpha) ---")
    print(analysis)
    
    print("\n--- Absolute Return Analysis ---")
    abs_analysis = risk_analysis(report_normal['return'])
    print(abs_analysis)
    
    # 保存报告
    import os
    os.makedirs("outputs", exist_ok=True)
    report_normal.to_csv('outputs/qlib_backtest_report.csv')
    print("\n✅ Backtest report saved to outputs/qlib_backtest_report.csv")

if __name__ == "__main__":
    run_backtest()
