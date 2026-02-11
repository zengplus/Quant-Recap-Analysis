import json
import os
import random
import warnings
import logging
import numpy as np
import torch
from model_core.config import ModelConfig
from model_core.qlib_loader import QlibDataLoader
from model_core.ashare_engine import AShareAlphaEngine
import pandas as pd
from model_core.run_backtest import run_joinquant_backtest, run_qlib_backtest, backtest_summary_row
from model_core.signal_engine import compute_signal_matrix
from model_core.vm import StackVM


def train_ashare(
    train_start,
    train_end,
    valid_start,
    valid_end,
    valid_periods=None,
    valid_agg="mean",
    instruments='csi300',
    batch_size=512,
    train_steps=500,
    max_formula_len=12,
    use_lord_regularization=True,
    lord_decay_rate=1e-3,
    lord_num_iterations=5,
    output_dir="outputs",
    provider_uri='/Users/shuyan/.qlib/qlib_data/cn_data/qlib_bin',
    seed=None,
    token_mode="postfix",
):
    _silence_noise()
    _configure_torch_threads()
    seed = _init_seed(seed)
    print(f"Seed: {seed}")
    ModelConfig.BATCH_SIZE = batch_size
    ModelConfig.TRAIN_STEPS = train_steps
    ModelConfig.MAX_FORMULA_LEN = max_formula_len
    train_loader = QlibDataLoader(provider_uri=provider_uri)
    # 增加 lookback_days 缓冲，防止 Rolling Window 导致训练初期全为 0
    # 默认 730 天，确保足够覆盖 120 天的窗口
    train_load_start = (pd.to_datetime(train_start) - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    train_loader.load_data(start_time=train_load_start, end_time=train_end, instruments=instruments, verbose=False, price_mode="pre_adjusted")
    
    # --- 关键修正：切片去除缓冲数据 ---
    # 我们加载了额外 730 天数据用于计算 Rolling Window 特征，
    # 但训练时只应关注 train_start 之后的表现。
    #如果不切片，每次评估都会跑多余的 730 天，严重拖慢速度且浪费显存。
    def slice_loader(loader, start_date):
        start_dt = pd.Timestamp(start_date)
        if loader.dates is None or len(loader.dates) == 0: return
        
        # searchsorted returns index of first element >= start_dt
        idx = loader.dates.searchsorted(start_dt)
        if idx == 0: return 
        
        print(f"Slicing loader: removing first {idx} days (Buffer period) to restore speed...")
        
        # Slice dates
        loader.dates = loader.dates[idx:]
        
        # Slice feat_tensor: (Assets, Features, Time) -> dim 2
        if loader.feat_tensor is not None:
            loader.feat_tensor = loader.feat_tensor[:, :, idx:]
            
        # Slice target_ret: (Assets, Time) -> dim 1
        if loader.target_ret is not None:
            loader.target_ret = loader.target_ret[:, idx:]
            
        # Slice raw_data_cache: (Assets, Time) -> dim 1
        if loader.raw_data_cache is not None:
            for k, v in loader.raw_data_cache.items():
                if isinstance(v, torch.Tensor):
                    if v.dim() == 2:
                        loader.raw_data_cache[k] = v[:, idx:]
                    elif v.dim() == 1:
                        loader.raw_data_cache[k] = v[idx:]
    
    slice_loader(train_loader, train_start)
    
    valid_loaders = []
    if valid_periods is None:
        valid_periods = [(valid_start, valid_end)]
    for vs, ve in list(valid_periods):
        vl = QlibDataLoader(provider_uri=provider_uri)
        valid_load_start = (pd.to_datetime(vs) - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
        vl.load_data(start_time=valid_load_start, end_time=ve, instruments=instruments, verbose=False, price_mode="pre_adjusted")
        slice_loader(vl, vs)
        valid_loaders.append(vl)
    engine = AShareAlphaEngine(
        train_loader,
        valid_loader=None,
        valid_loaders=valid_loaders,
        valid_agg=valid_agg,
        use_lord_regularization=use_lord_regularization,
        lord_decay_rate=lord_decay_rate,
        lord_num_iterations=lord_num_iterations,
        token_mode=token_mode,
    )
    engine.train()
    if engine.best_formula is None:
        engine.best_formula = [0]
    vm = StackVM()
    if str(token_mode).lower() == "prefix":
        engine.best_formula = vm.repair_prefix(engine.best_formula)
    else:
        engine.best_formula = vm.repair_postfix(engine.best_formula)
    os.makedirs(output_dir, exist_ok=True)
    seed_path = os.path.join(output_dir, "seed.json")
    with open(seed_path, "w") as f:
        json.dump({"seed": seed}, f)
    strategy_path = os.path.join(output_dir, "best_ashare_strategy.json")
    with open(strategy_path, "w") as f:
        json.dump(engine.best_formula, f)
    weights_path = os.path.join(output_dir, "alphagpt_model.pth")
    torch.save(engine.model.state_dict(), weights_path, _use_new_zipfile_serialization=False)
    return engine.best_formula, strategy_path, weights_path


def run_full_pipeline(
    train_start,
    train_end,
    valid_start,
    valid_end,
    backtest_start,
    backtest_end,
    instruments='csi300',
    batch_size=512,
    train_steps=1000,
    max_formula_len=12,
    output_dir="outputs",
    provider_uri='/Users/shuyan/.qlib/qlib_data/cn_data/qlib_bin',
    seed=None,
    token_mode="postfix",
    backtest_mode="joinquant_like",
):
    _silence_noise()
    _configure_torch_threads()
    formula, strategy_path, _ = train_ashare(
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        instruments=instruments,
        batch_size=batch_size,
        train_steps=train_steps,
        max_formula_len=max_formula_len,
        output_dir=output_dir,
        provider_uri=provider_uri,
        seed=seed,
        token_mode=token_mode,
    )
    signal_start = pd.to_datetime(backtest_start) - pd.Timedelta(days=220)
    signal_df = compute_signal_matrix(
        formula=formula,
        strategy_path=strategy_path,
        provider_uri=provider_uri,
        instruments=instruments,
        start_time=signal_start.strftime("%Y-%m-%d"),
        end_time=backtest_end,
        lookback_days=730,
        token_mode=token_mode,
        price_mode="pre_adjusted",
        monthly_resample=False,
    )[0]

    if backtest_mode == "qlib":
        report_df, positions_df, pred_df, analysis, abs_analysis = run_qlib_backtest(
            signal_df=signal_df,
            start_time=backtest_start,
            end_time=backtest_end,
            instruments=instruments,
            topk=getattr(ModelConfig, "TOPK", 10),
            benchmark="SH000300",
            provider_uri=provider_uri,
            price_mode="pre_adjusted",
            trade_price_mode="raw",
            ref_price_mode="raw",
            benchmark_price_mode="raw",
        )
        return formula, report_df, positions_df, pred_df, analysis, abs_analysis

    report_df, positions_df, pred_df, analysis, abs_analysis = run_joinquant_backtest(
        signal_df=signal_df,
        start_time=backtest_start,
        end_time=backtest_end,
        instruments=instruments,
        topk=getattr(ModelConfig, "TOPK", 10),
        benchmark="SH000300",
        provider_uri=provider_uri,
        signal_lag=1,
        slippage_rate=0.0003,
        trade_price_mode="raw",
        ref_price_mode="raw",
        benchmark_price_mode="raw",
        backfill_untradable=True,
        target_value_per_stock=None,
        corporate_action_mode="split",
    )
    return formula, report_df, positions_df, pred_df, analysis, abs_analysis


def run_multi_backtest(
    formula,
    periods,
    instruments='csi300',
    provider_uri='/Users/shuyan/.qlib/qlib_data/cn_data/qlib_bin',
    backtest_mode="cross_validate",
    token_mode="postfix",
):
    _silence_noise()
    _configure_torch_threads()
    rows = []
    for label, start_time, end_time in periods:
        signal_start = pd.to_datetime(start_time) - pd.Timedelta(days=220)
        signal_df, _ = compute_signal_matrix(
            formula=formula,
            provider_uri=provider_uri,
            instruments=instruments,
            start_time=signal_start.strftime("%Y-%m-%d"),
            end_time=end_time,
            lookback_days=730,
            token_mode=token_mode,
            price_mode="pre_adjusted",
            monthly_resample=False,
        )
        if backtest_mode == "joinquant_like":
            report_df, _, _, analysis, _ = run_joinquant_backtest(
                signal_df=signal_df,
                start_time=start_time,
                end_time=end_time,
                instruments=instruments,
                topk=getattr(ModelConfig, "TOPK", 10),
                benchmark="SH000300",
                provider_uri=provider_uri,
                signal_lag=1,
                slippage_rate=0.0003,
                trade_price_mode="raw",
                ref_price_mode="raw",
                benchmark_price_mode="raw",
                backfill_untradable=True,
                target_value_per_stock=None,
                corporate_action_mode="split",
            )
            row = backtest_summary_row(label, analysis, report_df)
            row["Backtest"] = "joinquant_like"
            row["CrossCheck"] = ""
            rows.append(row)
            continue

        if backtest_mode == "qlib":
            report_df, _, _, analysis, _ = run_qlib_backtest(
                signal_df=signal_df,
                start_time=start_time,
                end_time=end_time,
                instruments=instruments,
                topk=getattr(ModelConfig, "TOPK", 10),
                benchmark="SH000300",
                provider_uri=provider_uri,
                price_mode="pre_adjusted",
                trade_price_mode="raw",
                ref_price_mode="raw",
                benchmark_price_mode="raw",
            )
            row = backtest_summary_row(label, analysis, report_df)
            row["Backtest"] = "qlib"
            row["CrossCheck"] = ""
            rows.append(row)
            continue

        if backtest_mode != "cross_validate":
            raise ValueError(f"Unsupported backtest_mode: {backtest_mode}")

        report_jq, _, _, analysis_jq, _ = run_joinquant_backtest(
            signal_df=signal_df,
            start_time=start_time,
            end_time=end_time,
            instruments=instruments,
            topk=getattr(ModelConfig, "TOPK", 10),
            benchmark="SH000300",
            provider_uri=provider_uri,
            signal_lag=1,
            slippage_rate=0.0003,
            trade_price_mode="raw",
            ref_price_mode="raw",
            benchmark_price_mode="raw",
            backfill_untradable=True,
            target_value_per_stock=None,
            corporate_action_mode="split",
        )
        report_qlib, _, _, analysis_qlib, _ = run_qlib_backtest(
            signal_df=signal_df,
            start_time=start_time,
            end_time=end_time,
            instruments=instruments,
            topk=getattr(ModelConfig, "TOPK", 10),
            benchmark="SH000300",
            provider_uri=provider_uri,
            price_mode="pre_adjusted",
            trade_price_mode="raw",
            ref_price_mode="raw",
            benchmark_price_mode="raw",
        )

        row_jq = backtest_summary_row(label, analysis_jq, report_jq)
        row_qlib = backtest_summary_row(label, analysis_qlib, report_qlib)
        consistent = _row_consistent(row_jq, row_qlib)

        if consistent:
            row = dict(row_jq)
            row["Backtest"] = "both"
            row["CrossCheck"] = "一致"
            rows.append(row)
        else:
            r1 = dict(row_jq)
            r1["Backtest"] = "joinquant_like"
            r1["CrossCheck"] = "不一致"
            rows.append(r1)
            r2 = dict(row_qlib)
            r2["Backtest"] = "qlib"
            r2["CrossCheck"] = "不一致"
            rows.append(r2)
    return pd.DataFrame(rows)


def _init_seed(seed=None):
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _configure_torch_threads():
    try:
        n = os.cpu_count() or 1
        torch.set_num_threads(int(n))
    except Exception:
        pass


def _silence_noise():
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    logging.getLogger("qlib").setLevel(logging.ERROR)
    logging.getLogger("qlib").propagate = False
    logging.getLogger("qlib.backtest").setLevel(logging.ERROR)
    logging.getLogger("qlib.backtest").propagate = False
    logging.getLogger("qlib.backtest.exchange").setLevel(logging.ERROR)
    logging.getLogger("qlib.backtest.exchange").propagate = False
    logging.getLogger("gym").setLevel(logging.ERROR)


def _row_consistent(row_a, row_b, pct_atol=1e-3, ir_atol=1e-6, ir_rtol=1e-6):
    keys_pct = ("TotalReturn%", "BenchReturn%", "ExcessReturn%", "MaxDD%")
    for k in keys_pct:
        a = float(row_a.get(k, 0.0))
        b = float(row_b.get(k, 0.0))
        if abs(a - b) > float(pct_atol):
            return False
    a_ir = float(row_a.get("IR", 0.0))
    b_ir = float(row_b.get("IR", 0.0))
    if abs(a_ir - b_ir) > float(ir_atol) + float(ir_rtol) * abs(b_ir):
        return False
    return True


def main():
    import datetime
    import time
    
    # 获取当前时间戳作为本次批量实验的ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    
    print(f"Starting 1 experiment (Batch ID: {timestamp})...")
    
    for i in range(1):
        print(f"\n{'='*20} Running Experiment {i+1}/1 {'='*20}")
        # 为每次运行创建一个独立的带时间戳的目录
        run_id = f"{timestamp}_run_{i+1:02d}"
        output_dir = os.path.join("outputs", f"experiment_{run_id}")
        
        # 1. 运行全流程：训练 -> 验证 -> 测试回测
        # 注意：seed=None 表示每次使用随机种子，保证结果多样性
        formula, *_ = run_full_pipeline(
            train_start='2018-01-01',
            train_end='2023-12-31',
            valid_start='2024-01-01',
            valid_end='2024-12-31',
            backtest_start='2025-01-01',
            backtest_end='2025-12-31',
            instruments='csi300',
            batch_size=512,
            train_steps=200,  # Increased slightly for better convergence
            max_formula_len=12,
            output_dir=output_dir,
            seed=None 
        )
        
        # 2. 运行多期回测（Train/Valid/Test）以获取完整评估指标
        periods = [
            ("train", '2018-01-01', '2023-12-31'),
            ("valid", '2024-01-01', '2024-12-31'),
            ("test", '2025-01-01', '2025-12-31')
        ]
        summary_df = run_multi_backtest(formula, periods, instruments='csi300')
        
        # 3. 保存本次运行的详细 summary
        summary_path = os.path.join(output_dir, "backtest_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Formula {i+1}: {formula}")
        print(summary_df)
        
        # 4. 收集关键指标用于最终汇总
        # 提取 Valid 和 Test 集的 IC, IR, Return 等
        res = {"run_id": run_id, "formula": str(formula)}
        for _, row in summary_df.iterrows():
            group = row['label'] # 'train', 'valid', 'test'
            # 自动提取所有数值列
            for col in row.index:
                if col not in ['label', 'Backtest', 'CrossCheck']:
                    res[f"{group}_{col}"] = row[col]
        all_results.append(res)

    # 5. 实验结束，保存总表
    final_df = pd.DataFrame(all_results)
    final_summary_path = os.path.join("outputs", f"batch_summary_{timestamp}.csv")
    final_df.to_csv(final_summary_path, index=False)
    
    print("\n" + "="*50)
    print("All 5 Experiments Completed.")
    print(f"Final summary saved to: {final_summary_path}")
    print("Top results sorted by Test_IR:")
    # 如果有 Test_IR 列，按其排序显示
    if 'test_IR' in final_df.columns:
        print(final_df.sort_values(by='test_IR', ascending=False)[['run_id', 'formula', 'test_IR', 'test_TotalReturn%', 'valid_IR']].to_string())
    else:
        print(final_df)


if __name__ == "__main__":
    main()
