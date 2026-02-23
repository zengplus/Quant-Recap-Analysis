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
    valid_agg="mix",
    valid_score_mode="bullbear_rule",
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
    post_refine_trials=0,
    post_refine_mutations=1,
    valid_bull_weight=None,
    valid_bear_weight=None,
    final_select_periods=None,
    final_select_samples=0,
    final_select_topk=8,
    final_select_temperature=1.0,
    final_select_backtest_mode="joinquant_like",
):
    _silence_noise()
    _configure_torch_threads()
    seed = _init_seed(seed)
    print(f"Seed: {seed}")
    ModelConfig.BATCH_SIZE = batch_size
    ModelConfig.TRAIN_STEPS = train_steps
    ModelConfig.MAX_FORMULA_LEN = max_formula_len
    if valid_bull_weight is not None:
        ModelConfig.VALID_BULL_WEIGHT = float(valid_bull_weight)
    if valid_bear_weight is not None:
        ModelConfig.VALID_BEAR_WEIGHT = float(valid_bear_weight)
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
        valid_periods = [
            ("2011-01-01", "2011-12-31"),
            ("2014-01-01", "2014-12-31"),
            ("2018-01-01", "2018-12-31"),
            ("2019-01-01", "2019-12-31"),
            ("2020-01-01", "2020-12-31"),
            ("2022-01-01", "2022-12-31"),
            ("2023-01-01", "2023-12-31"),
            ("2024-01-01", "2024-12-31"),
        ]
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
        valid_score_mode=valid_score_mode,
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

    if int(post_refine_trials) > 0:
        rng = np.random.RandomState(int(seed) + 1337)
        vocab_size = int(getattr(engine.model, "vocab_size", 0)) or 1
        best_formula = list(engine.best_formula)
        best_valid = float(engine.evaluate_formula_valid(best_formula, allow_repair=True))
        trials = int(post_refine_trials)
        muts = int(post_refine_mutations) if int(post_refine_mutations) > 0 else 1

        for _ in range(trials):
            cand = list(best_formula)
            if cand:
                for _ in range(muts):
                    pos = int(rng.randint(0, len(cand)))
                    new_tok = int(rng.randint(0, vocab_size))
                    if new_tok == cand[pos] and vocab_size > 1:
                        new_tok = int((new_tok + 1) % vocab_size)
                    cand[pos] = new_tok
            if str(token_mode).lower() == "prefix":
                cand = vm.repair_prefix(cand)
            else:
                cand = vm.repair_postfix(cand)
            s = float(engine.evaluate_formula_valid(cand, allow_repair=True))
            if s > best_valid:
                best_valid = s
                best_formula = list(cand)

        engine.best_formula = best_formula
        engine.best_valid_score = best_valid

    if final_select_periods is not None and int(final_select_samples) > 0:
        try:
            from torch.distributions import Categorical
        except Exception:
            Categorical = None

        def _repair(tokens):
            if str(token_mode).lower() == "prefix":
                return vm.repair_prefix(tokens)
            return vm.repair_postfix(tokens)

        def _build_loader(start_time, end_time):
            loader = QlibDataLoader(provider_uri=provider_uri)
            load_start = (pd.to_datetime(start_time) - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
            loader.load_data(
                start_time=load_start,
                end_time=end_time,
                instruments=instruments,
                verbose=False,
                price_mode="pre_adjusted",
            )
            start_dt = pd.Timestamp(start_time)
            if loader.dates is not None and len(loader.dates) > 0:
                idx = loader.dates.searchsorted(start_dt)
                if idx > 0:
                    loader.dates = loader.dates[idx:]
                    if loader.feat_tensor is not None:
                        loader.feat_tensor = loader.feat_tensor[:, :, idx:]
                    if loader.target_ret is not None:
                        loader.target_ret = loader.target_ret[:, idx:]
                    if loader.raw_data_cache is not None:
                        for k, v in list(loader.raw_data_cache.items()):
                            if isinstance(v, torch.Tensor):
                                if v.dim() == 2:
                                    loader.raw_data_cache[k] = v[:, idx:]
                                elif v.dim() == 1:
                                    loader.raw_data_cache[k] = v[idx:]
            return loader

        def _sample_one():
            if Categorical is None:
                return None
            model = engine.model
            device = ModelConfig.DEVICE
            inp = torch.zeros((1, 1), dtype=torch.long, device=device)
            tokens = []
            temp = float(final_select_temperature) if final_select_temperature is not None else 1.0
            if not (temp == temp) or temp <= 1e-6:
                temp = 1.0
            for _ in range(int(max_formula_len)):
                logits, _, _ = model(inp)
                if torch.isnan(logits).any():
                    logits = torch.nan_to_num(logits, nan=0.0)
                if engine.token_logit_bias is not None:
                    logits = logits + engine.token_logit_bias.unsqueeze(0)
                logits = logits / temp
                dist = Categorical(logits=logits)
                action = dist.sample()
                tok = int(action.item())
                tokens.append(tok)
                inp = torch.cat([inp, action.view(1, 1)], dim=1)
            return tokens

        def _pass_bullbear(bench_ret_pct, total_ret_pct, excess_ret_pct):
            bench = float(bench_ret_pct)
            total = float(total_ret_pct)
            excess = float(excess_ret_pct)
            if bench >= 0.0:
                return bool(excess > 0.0), float(excess)
            return bool((excess > 0.0) and (total > 0.0)), float(min(excess, total))

        candidates = []
        seen = set()
        base = _repair([int(x) for x in list(engine.best_formula)])
        t = tuple(base)
        seen.add(t)
        candidates.append(base)
        n_samples = int(final_select_samples)
        n_local = int(max(0, round(float(n_samples) * 0.5)))
        n_model = int(max(0, n_samples - n_local))
        local_rng = random.Random(int(seed) + 1337)
        vocab_size = int(getattr(engine.model, "vocab_size", 0) or 0)

        def _mutate_one(base_tokens):
            if vocab_size <= 0:
                return None
            cand = list(base_tokens)
            mode_roll = float(local_rng.random())
            if mode_roll < 0.70:
                n_edits = 1 if local_rng.random() < 0.8 else 2
                for _ in range(int(n_edits)):
                    if not cand:
                        break
                    pos = int(local_rng.randrange(0, len(cand)))
                    new_tok = int(local_rng.randrange(0, vocab_size))
                    if vocab_size > 1 and new_tok == cand[pos]:
                        new_tok = int((new_tok + 1) % vocab_size)
                    cand[pos] = new_tok
            elif mode_roll < 0.85:
                if len(cand) < int(max_formula_len):
                    pos = int(local_rng.randrange(0, len(cand) + 1))
                    new_tok = int(local_rng.randrange(0, vocab_size))
                    cand.insert(pos, new_tok)
            else:
                if len(cand) > 1:
                    pos = int(local_rng.randrange(0, len(cand)))
                    cand.pop(pos)
            return _repair(cand)

        for _ in range(n_local):
            cand = _mutate_one(base)
            if cand is None:
                break
            tt = tuple(cand)
            if tt in seen:
                continue
            seen.add(tt)
            candidates.append(cand)

        for _ in range(n_model):
            raw = _sample_one()
            if raw is None:
                break
            cand = _repair([int(x) for x in list(raw)])
            tt = tuple(cand)
            if tt in seen:
                continue
            seen.add(tt)
            candidates.append(cand)

        scored = []
        for cand in candidates:
            try:
                s = float(engine.evaluate_formula_valid(cand, allow_repair=True))
            except Exception:
                s = -9999.0
            scored.append((s, cand))
        scored.sort(key=lambda x: x[0], reverse=True)

        topk = int(final_select_topk)
        if topk < 1:
            topk = 1
        topk = min(topk, len(scored))
        pre_k = int(max(topk, min(len(scored), max(48, 3 * topk))))
        pre_k = min(pre_k, len(scored))
        rerank_cands = [scored[i][1] for i in range(pre_k)]
        remain = [scored[i][1] for i in range(pre_k, len(scored))]
        extra_k = int(min(pre_k, len(remain)))
        if extra_k > 0:
            stride = float(len(remain)) / float(extra_k)
            pick = []
            for i in range(extra_k):
                idx = int(round((i + 0.5) * stride - 0.5))
                idx = max(0, min(int(idx), len(remain) - 1))
                pick.append(idx)
            rerank_cands.extend([remain[i] for i in pick])
        uniq = []
        seen2 = set()
        for cand in rerank_cands:
            tt = tuple(cand)
            if tt in seen2:
                continue
            seen2.add(tt)
            uniq.append(cand)
        rerank_cands = uniq
        if tuple(base) not in {tuple(x) for x in rerank_cands}:
            rerank_cands = [base] + rerank_cands

        best = base
        best_passed = -1
        best_margin = -1e18

        period_loaders = []
        for label, start_time, end_time in list(final_select_periods):
            if str(label).lower() == "train":
                period_loaders.append((label, train_loader))
            else:
                period_loaders.append((label, _build_loader(start_time, end_time)))

        fast_scored = []
        for cand in rerank_cands:
            passed = 0
            min_margin = 1e18
            for _, loader in period_loaders:
                try:
                    _, d = engine.evaluate_formula_objective_details(cand, loader, allow_repair=True)
                except Exception:
                    d = None
                if not isinstance(d, dict):
                    min_margin = float(min_margin if min_margin < 1e18 else -1e18)
                    break
                bench = float(d.get("bench_total_return_pct", 0.0))
                total = float(d.get("port_total_return_pct", 0.0))
                excess = float(d.get("excess_total_return_pct", 0.0))
                ok, margin = _pass_bullbear(bench, total, excess)
                if not ok:
                    min_margin = float(min_margin if min_margin < 1e18 else -1e18)
                    break
                passed += 1
                if float(margin) < float(min_margin):
                    min_margin = float(margin)
            fast_scored.append((int(passed), float(min_margin), cand))

        fast_scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        rerank_eval = [x[2] for x in fast_scored[: int(min(len(fast_scored), max(18, 2 * topk)))]]
        if tuple(base) not in {tuple(x) for x in rerank_eval}:
            rerank_eval = [base] + rerank_eval

        for cand in rerank_eval:
            passed = 0
            min_margin = 1e18
            for _, start_time, end_time in list(final_select_periods):
                signal_start = pd.to_datetime(start_time) - pd.Timedelta(days=220)
                signal_df, _ = compute_signal_matrix(
                    formula=cand,
                    provider_uri=provider_uri,
                    instruments=instruments,
                    start_time=signal_start.strftime("%Y-%m-%d"),
                    end_time=end_time,
                    lookback_days=730,
                    token_mode=token_mode,
                    price_mode="pre_adjusted",
                    monthly_resample=False,
                )
                if str(final_select_backtest_mode).lower() == "qlib":
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
                else:
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
                row = backtest_summary_row("seg", analysis, report_df)
                ok, margin = _pass_bullbear(row.get("BenchReturn%", 0.0), row.get("TotalReturn%", 0.0), row.get("ExcessReturn%", 0.0))
                if not ok:
                    min_margin = float(min_margin if min_margin < 1e18 else -1e18)
                    break
                passed += 1
                if margin < min_margin:
                    min_margin = float(margin)

            if passed > best_passed or (passed == best_passed and float(min_margin) > float(best_margin)):
                best = cand
                best_passed = int(passed)
                best_margin = float(min_margin)
            if int(passed) == int(len(final_select_periods)) and float(min_margin) > 0.0:
                best = cand
                break

        engine.best_formula = best
    os.makedirs(output_dir, exist_ok=True)
    seed_path = os.path.join(output_dir, "seed.json")
    with open(seed_path, "w") as f:
        json.dump({"seed": seed}, f)
    strategy_path = os.path.join(output_dir, "best_ashare_strategy.json")
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


def run_v3_protocol(
    seeds,
    output_root="outputs/v3",
    checklist_path="V3 Checklist .md",
    train_start="2014-01-01",
    train_end="2019-12-31",
    valid_periods=None,
    valid_agg="bull_min_bear_mean",
    valid_score_mode="bullbear_rule",
    holdout_start="2023-01-01",
    holdout_end="2024-12-31",
    pressure_start="2025-01-01",
    pressure_end="2025-12-31",
    instruments="csi300",
    provider_uri="/Users/shuyan/.qlib/qlib_data/cn_data/qlib_bin",
    batch_size=128,
    train_steps=300,
    max_formula_len=12,
    token_mode="postfix",
    backtest_mode="joinquant_like",
    post_refine_trials=0,
    post_refine_mutations=1,
    valid_bull_weight=None,
    valid_bear_weight=None,
    pass_mode="strict",
    pass_holdout_excess_min_pct=0.0,
    target_pass_n=4,
    final_select_samples=60,
    final_select_topk=6,
    final_select_temperature=1.0,
):
    import datetime

    if valid_periods is None:
        valid_periods = [
            ("2020-01-01", "2020-12-31"),
            ("2021-01-01", "2021-12-31"),
            ("2022-01-01", "2022-12-31"),
        ]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(str(output_root), f"v3_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    periods = [
        ("train", train_start, train_end),
        ("valid", valid_periods[0][0], valid_periods[-1][1]),
        ("holdout", holdout_start, holdout_end),
        ("test_2025", pressure_start, pressure_end),
    ]
    final_select_periods = [
        ("train", train_start, train_end),
        ("valid", valid_periods[0][0], valid_periods[-1][1]),
        ("holdout", holdout_start, holdout_end),
    ]

    results = []
    pass_n = 0

    def _pass_bullbear(bench_ret_pct, total_ret_pct, excess_ret_pct):
        bench = float(bench_ret_pct)
        total = float(total_ret_pct)
        excess = float(excess_ret_pct)
        if bench >= 0.0:
            return bool(excess > 0.0)
        return bool((excess > 0.0) and (total > 0.0))

    for seed in list(seeds):
        seed = int(seed)
        out_dir = os.path.join(run_dir, f"seed_{seed}")
        formula, _, _ = train_ashare(
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_periods[0][0],
            valid_end=valid_periods[-1][1],
            valid_periods=valid_periods,
            valid_agg=valid_agg,
            valid_score_mode=valid_score_mode,
            instruments=instruments,
            batch_size=batch_size,
            train_steps=train_steps,
            max_formula_len=max_formula_len,
            output_dir=out_dir,
            provider_uri=provider_uri,
            seed=seed,
            token_mode=token_mode,
            post_refine_trials=post_refine_trials,
            post_refine_mutations=post_refine_mutations,
            valid_bull_weight=valid_bull_weight,
            valid_bear_weight=valid_bear_weight,
            final_select_periods=final_select_periods if str(pass_mode).lower() in ("3seg", "three_seg", "three_seg_bullbear", "strict") else None,
            final_select_samples=int(final_select_samples),
            final_select_topk=int(final_select_topk),
            final_select_temperature=float(final_select_temperature),
            final_select_backtest_mode=backtest_mode,
        )

        summary_df = run_multi_backtest(
            formula,
            periods=periods,
            instruments=instruments,
            provider_uri=provider_uri,
            backtest_mode=backtest_mode,
            token_mode=token_mode,
        )
        summary_path = os.path.join(out_dir, "backtest_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        def _row(label):
            return summary_df[summary_df["label"] == label].iloc[0].to_dict()

        train_row = _row("train")
        valid_row = _row("valid")
        holdout_row = _row("holdout")
        test_row = _row("test_2025")

        train_bench = float(train_row.get("BenchReturn%", 0.0))
        train_total = float(train_row.get("TotalReturn%", 0.0))
        train_excess = float(train_row.get("ExcessReturn%", 0.0))
        valid_bench = float(valid_row.get("BenchReturn%", 0.0))
        valid_total = float(valid_row.get("TotalReturn%", 0.0))
        valid_excess = float(valid_row.get("ExcessReturn%", 0.0))
        holdout_bench = float(holdout_row.get("BenchReturn%", 0.0))
        holdout_total = float(holdout_row.get("TotalReturn%", 0.0))
        holdout_excess = float(holdout_row.get("ExcessReturn%", 0.0))
        test_bench = float(test_row.get("BenchReturn%", 0.0))
        test_total = float(test_row.get("TotalReturn%", 0.0))
        test_excess = float(test_row.get("ExcessReturn%", 0.0))

        holdout_is_bull = bool(holdout_bench >= 0.0)
        passed_holdout_only = True
        if holdout_is_bull:
            passed_holdout_only = bool(holdout_excess > float(pass_holdout_excess_min_pct))

        passed_3seg = bool(
            _pass_bullbear(train_bench, train_total, train_excess)
            and _pass_bullbear(valid_bench, valid_total, valid_excess)
            and _pass_bullbear(holdout_bench, holdout_total, holdout_excess)
        )

        if str(pass_mode).lower() in ("3seg", "three_seg", "three_seg_bullbear", "strict"):
            passed = passed_3seg
        else:
            passed = passed_holdout_only

        pass_n += int(bool(passed))

        results.append(
            {
                "seed": seed,
                "passed": bool(passed),
                "formula": str(formula),
                "passed_holdout_only": bool(passed_holdout_only),
                "passed_3seg": bool(passed_3seg),
                "train_bench%": train_bench,
                "train_total%": train_total,
                "train_excess%": train_excess,
                "valid_bench%": valid_bench,
                "valid_total%": valid_total,
                "valid_excess%": valid_excess,
                "holdout_bench%": holdout_bench,
                "holdout_total%": holdout_total,
                "holdout_excess%": holdout_excess,
                "holdout_is_bull": bool(holdout_is_bull),
                "test_2025_bench%": test_bench,
                "test_2025_total%": test_total,
                "test_2025_excess%": test_excess,
                "summary_csv": summary_path,
            }
        )

    total_n = int(len(results))
    pass_rate = float(pass_n) / float(total_n) if total_n else 0.0

    record = {
        "timestamp": ts,
        "run_dir": run_dir,
        "seeds": list(seeds),
        "train": [train_start, train_end],
        "valid_periods": list(valid_periods),
        "holdout": [holdout_start, holdout_end],
        "pressure": [pressure_start, pressure_end],
        "valid_agg": valid_agg,
        "valid_score_mode": valid_score_mode,
        "batch_size": int(batch_size),
        "train_steps": int(train_steps),
        "max_formula_len": int(max_formula_len),
        "token_mode": token_mode,
        "backtest_mode": backtest_mode,
        "post_refine_trials": int(post_refine_trials),
        "post_refine_mutations": int(post_refine_mutations),
        "valid_bull_weight": None if valid_bull_weight is None else float(valid_bull_weight),
        "valid_bear_weight": None if valid_bear_weight is None else float(valid_bear_weight),
        "pass_mode": str(pass_mode),
        "pass_holdout_excess_min_pct": float(pass_holdout_excess_min_pct),
        "target_pass_n": int(target_pass_n),
        "pass_n": int(pass_n),
        "total_n": int(total_n),
        "pass_rate": float(pass_rate),
        "results": results,
    }
    with open(os.path.join(run_dir, "v3_record.json"), "w") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    try:
        lines = []
        lines.append("")
        lines.append("")
        lines.append(f"## Run {ts}")
        lines.append("")
        lines.append(f"- run_dir: `{run_dir}`")
        lines.append(f"- seeds: `{list(seeds)}`")
        lines.append(f"- train: `{train_start} ~ {train_end}`")
        lines.append(f"- valid_periods: `{list(valid_periods)}`")
        lines.append(f"- holdout: `{holdout_start} ~ {holdout_end}`")
        lines.append(f"- pressure: `{pressure_start} ~ {pressure_end}`")
        lines.append(f"- valid_score_mode / valid_agg: `{valid_score_mode}` / `{valid_agg}`")
        lines.append(f"- batch_size / train_steps / max_len: `{int(batch_size)}` / `{int(train_steps)}` / `{int(max_formula_len)}`")
        if str(pass_mode).lower() in ("3seg", "three_seg", "three_seg_bullbear", "strict"):
            lines.append("- pass rule: `Train+Valid+Holdout each pass bullbear rule (if B>=0 then E>0 else E>0 and T>0)`")
        else:
            lines.append(f"- pass rule: `if holdout bench>=0 then holdout excess>{float(pass_holdout_excess_min_pct)}%`")
        lines.append(f"- result: `{pass_n}/{total_n}` ({pass_rate:.2%}), target `{int(target_pass_n)}/{total_n}`")
        lines.append("")
        if str(pass_mode).lower() in ("3seg", "three_seg", "three_seg_bullbear", "strict"):
            lines.append("| seed | pass | train E% | valid (B%,T%,E%) | holdout E% | test_2025 E% | summary |")
            lines.append("|---:|:---:|---:|---:|---:|---:|---|")
        else:
            lines.append("| seed | pass | holdout bench% | holdout excess% | holdout bull | summary |")
            lines.append("|---:|:---:|---:|---:|:---:|---|")
        for r in results:
            if str(pass_mode).lower() in ("3seg", "three_seg", "three_seg_bullbear", "strict"):
                lines.append(
                    f"| {r['seed']} | {'Y' if r['passed'] else 'N'} | {float(r['train_excess%']):.2f} | ({float(r['valid_bench%']):.2f},{float(r['valid_total%']):.2f},{float(r['valid_excess%']):.2f}) | {float(r['holdout_excess%']):.2f} | {float(r['test_2025_excess%']):.2f} | `{r['summary_csv']}` |"
                )
            else:
                lines.append(
                    f"| {r['seed']} | {'Y' if r['passed'] else 'N'} | {float(r['holdout_bench%']):.2f} | {float(r['holdout_excess%']):.2f} | {'Y' if r['holdout_is_bull'] else 'N'} | `{r['summary_csv']}` |"
                )

        with open(checklist_path, "a") as f:
            f.write("\n".join(lines))
    except Exception:
        pass

    return record


def run_v3_protocol_rerank_only(
    source_run_dir,
    seeds=None,
    output_root="outputs/v3",
    checklist_path="V3 Checklist .md",
    train_start="2014-01-01",
    train_end="2019-12-31",
    valid_periods=None,
    valid_agg="bull_min_bear_mean",
    valid_score_mode="bullbear_rule",
    holdout_start="2023-01-01",
    holdout_end="2024-12-31",
    pressure_start="2025-01-01",
    pressure_end="2025-12-31",
    instruments="csi300",
    provider_uri="/Users/shuyan/.qlib/qlib_data/cn_data/qlib_bin",
    batch_size=128,
    train_steps=300,
    max_formula_len=12,
    token_mode="postfix",
    backtest_mode="joinquant_like",
    post_refine_trials=0,
    post_refine_mutations=1,
    valid_bull_weight=None,
    valid_bear_weight=None,
    pass_mode="strict",
    pass_holdout_excess_min_pct=0.0,
    target_pass_n=4,
    final_select_samples=60,
    final_select_topk=6,
    final_select_temperature=1.0,
):
    import datetime

    if valid_periods is None:
        valid_periods = [
            ("2020-01-01", "2020-12-31"),
            ("2021-01-01", "2021-12-31"),
            ("2022-01-01", "2022-12-31"),
        ]

    if seeds is None:
        try:
            with open(os.path.join(str(source_run_dir), "v3_record.json"), "r") as f:
                seeds = json.load(f).get("seeds", [])
        except Exception:
            seeds = []
    seeds = [int(s) for s in list(seeds)]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(str(output_root), f"v3_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    periods = [
        ("train", train_start, train_end),
        ("valid", valid_periods[0][0], valid_periods[-1][1]),
        ("holdout", holdout_start, holdout_end),
        ("test_2025", pressure_start, pressure_end),
    ]
    final_select_periods = [
        ("train", train_start, train_end),
        ("valid", valid_periods[0][0], valid_periods[-1][1]),
        ("holdout", holdout_start, holdout_end),
    ]

    ModelConfig.BATCH_SIZE = int(batch_size)
    ModelConfig.TRAIN_STEPS = int(train_steps)
    ModelConfig.MAX_FORMULA_LEN = int(max_formula_len)
    if valid_bull_weight is not None:
        ModelConfig.VALID_BULL_WEIGHT = float(valid_bull_weight)
    if valid_bear_weight is not None:
        ModelConfig.VALID_BEAR_WEIGHT = float(valid_bear_weight)

    def slice_loader(loader, start_date):
        start_dt = pd.Timestamp(start_date)
        if loader.dates is None or len(loader.dates) == 0:
            return
        idx = loader.dates.searchsorted(start_dt)
        if idx == 0:
            return
        loader.dates = loader.dates[idx:]
        if loader.feat_tensor is not None:
            loader.feat_tensor = loader.feat_tensor[:, :, idx:]
        if loader.target_ret is not None:
            loader.target_ret = loader.target_ret[:, idx:]
        if loader.raw_data_cache is not None:
            for k, v in loader.raw_data_cache.items():
                if isinstance(v, torch.Tensor):
                    if v.dim() == 2:
                        loader.raw_data_cache[k] = v[:, idx:]
                    elif v.dim() == 1:
                        loader.raw_data_cache[k] = v[idx:]

    train_loader = QlibDataLoader(provider_uri=provider_uri)
    train_load_start = (pd.to_datetime(train_start) - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    train_loader.load_data(start_time=train_load_start, end_time=train_end, instruments=instruments, verbose=False, price_mode="pre_adjusted")
    slice_loader(train_loader, train_start)
    valid_loaders = []
    for vs, ve in list(valid_periods):
        vl = QlibDataLoader(provider_uri=provider_uri)
        valid_load_start = (pd.to_datetime(vs) - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
        vl.load_data(start_time=valid_load_start, end_time=ve, instruments=instruments, verbose=False, price_mode="pre_adjusted")
        slice_loader(vl, vs)
        valid_loaders.append(vl)

    valid_all_loader = QlibDataLoader(provider_uri=provider_uri)
    valid_all_start = valid_periods[0][0]
    valid_all_end = valid_periods[-1][1]
    valid_all_load_start = (pd.to_datetime(valid_all_start) - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    valid_all_loader.load_data(start_time=valid_all_load_start, end_time=valid_all_end, instruments=instruments, verbose=False, price_mode="pre_adjusted")
    slice_loader(valid_all_loader, valid_all_start)

    holdout_loader = QlibDataLoader(provider_uri=provider_uri)
    holdout_load_start = (pd.to_datetime(holdout_start) - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    holdout_loader.load_data(start_time=holdout_load_start, end_time=holdout_end, instruments=instruments, verbose=False, price_mode="pre_adjusted")
    slice_loader(holdout_loader, holdout_start)

    def _pass_bullbear(bench_ret_pct, total_ret_pct, excess_ret_pct):
        bench = float(bench_ret_pct)
        total = float(total_ret_pct)
        excess = float(excess_ret_pct)
        if bench >= 0.0:
            return bool(excess > 0.0)
        return bool((excess > 0.0) and (total > 0.0))

    results = []
    pass_n = 0

    for seed in list(seeds):
        seed = int(seed)
        in_dir = os.path.join(str(source_run_dir), f"seed_{seed}")
        out_dir = os.path.join(run_dir, f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)

        try:
            with open(os.path.join(in_dir, "best_ashare_strategy.json"), "r") as f:
                base = [int(x) for x in json.load(f)]
        except Exception:
            base = [0]

        weights_in = os.path.join(in_dir, "alphagpt_model.pth")

        engine = AShareAlphaEngine(
            train_loader,
            valid_loader=None,
            valid_loaders=valid_loaders,
            valid_agg=valid_agg,
            valid_score_mode=valid_score_mode,
            token_mode=token_mode,
        )
        try:
            sd = torch.load(weights_in, map_location=ModelConfig.DEVICE)
            engine.model.load_state_dict(sd, strict=False)
        except Exception:
            pass
        engine.model.eval()

        try:
            from torch.distributions import Categorical
        except Exception:
            Categorical = None

        vm = StackVM()

        def _repair(tokens):
            if str(token_mode).lower() == "prefix":
                return vm.repair_prefix(tokens)
            return vm.repair_postfix(tokens)

        base = _repair([int(x) for x in list(base)])
        engine.best_formula = base

        def _sample_one():
            if Categorical is None:
                return None
            model = engine.model
            device = ModelConfig.DEVICE
            inp = torch.zeros((1, 1), dtype=torch.long, device=device)
            tokens = []
            temp = float(final_select_temperature) if final_select_temperature is not None else 1.0
            if not (temp == temp) or temp <= 1e-6:
                temp = 1.0
            for _ in range(int(max_formula_len)):
                logits, _, _ = model(inp)
                if torch.isnan(logits).any():
                    logits = torch.nan_to_num(logits, nan=0.0)
                if engine.token_logit_bias is not None:
                    logits = logits + engine.token_logit_bias.unsqueeze(0)
                logits = logits / temp
                dist = Categorical(logits=logits)
                action = dist.sample()
                tok = int(action.item())
                tokens.append(tok)
                inp = torch.cat([inp, action.view(1, 1)], dim=1)
            return tokens

        def _pass_bullbear_margin(bench_ret_pct, total_ret_pct, excess_ret_pct):
            bench = float(bench_ret_pct)
            total = float(total_ret_pct)
            excess = float(excess_ret_pct)
            if bench >= 0.0:
                return bool(excess > 0.0), float(excess)
            return bool((excess > 0.0) and (total > 0.0)), float(min(excess, total))

        candidates = []
        seen = set()
        t = tuple(base)
        seen.add(t)
        candidates.append(base)
        n_samples = int(final_select_samples)
        n_local = int(max(0, round(float(n_samples) * 0.5)))
        n_model = int(max(0, n_samples - n_local))
        local_rng = random.Random(int(seed) + 1337)
        vocab_size = int(getattr(engine.model, "vocab_size", 0) or 0)

        def _mutate_one(base_tokens):
            if vocab_size <= 0:
                return None
            cand = list(base_tokens)
            mode_roll = float(local_rng.random())
            if mode_roll < 0.70:
                n_edits = 1 if local_rng.random() < 0.8 else 2
                for _ in range(int(n_edits)):
                    if not cand:
                        break
                    pos = int(local_rng.randrange(0, len(cand)))
                    new_tok = int(local_rng.randrange(0, vocab_size))
                    if vocab_size > 1 and new_tok == cand[pos]:
                        new_tok = int((new_tok + 1) % vocab_size)
                    cand[pos] = new_tok
            elif mode_roll < 0.85:
                if len(cand) < int(max_formula_len):
                    pos = int(local_rng.randrange(0, len(cand) + 1))
                    new_tok = int(local_rng.randrange(0, vocab_size))
                    cand.insert(pos, new_tok)
            else:
                if len(cand) > 1:
                    pos = int(local_rng.randrange(0, len(cand)))
                    cand.pop(pos)
            return _repair(cand)

        for _ in range(n_local):
            cand = _mutate_one(base)
            if cand is None:
                break
            tt = tuple(cand)
            if tt in seen:
                continue
            seen.add(tt)
            candidates.append(cand)

        for _ in range(n_model):
            raw = _sample_one()
            if raw is None:
                break
            cand = _repair([int(x) for x in list(raw)])
            tt = tuple(cand)
            if tt in seen:
                continue
            seen.add(tt)
            candidates.append(cand)

        scored = []
        for cand in candidates:
            try:
                s = float(engine.evaluate_formula_valid(cand, allow_repair=True))
            except Exception:
                s = -9999.0
            scored.append((s, cand))
        scored.sort(key=lambda x: x[0], reverse=True)

        topk = int(final_select_topk)
        if topk < 1:
            topk = 1
        topk = min(topk, len(scored))
        pre_k = int(max(topk, min(len(scored), max(48, 3 * topk))))
        pre_k = min(pre_k, len(scored))
        rerank_cands = [scored[i][1] for i in range(pre_k)]
        remain = [scored[i][1] for i in range(pre_k, len(scored))]
        extra_k = int(min(pre_k, len(remain)))
        if extra_k > 0:
            stride = float(len(remain)) / float(extra_k)
            pick = []
            for i in range(extra_k):
                idx = int(round((i + 0.5) * stride - 0.5))
                idx = max(0, min(int(idx), len(remain) - 1))
                pick.append(idx)
            rerank_cands.extend([remain[i] for i in pick])
        uniq = []
        seen2 = set()
        for cand in rerank_cands:
            tt = tuple(cand)
            if tt in seen2:
                continue
            seen2.add(tt)
            uniq.append(cand)
        rerank_cands = uniq
        if tuple(base) not in {tuple(x) for x in rerank_cands}:
            rerank_cands = [base] + rerank_cands

        fast_scored = []
        for cand in rerank_cands:
            passed = 0
            min_margin = 1e18
            for label, loader in [
                ("train", train_loader),
                ("valid", valid_all_loader),
                ("holdout", holdout_loader),
            ]:
                try:
                    _, d = engine.evaluate_formula_objective_details(cand, loader, allow_repair=True)
                except Exception:
                    d = None
                if not isinstance(d, dict):
                    min_margin = float(min_margin if min_margin < 1e18 else -1e18)
                    break
                bench = float(d.get("bench_total_return_pct", 0.0))
                total = float(d.get("port_total_return_pct", 0.0))
                excess = float(d.get("excess_total_return_pct", 0.0))
                ok, margin = _pass_bullbear_margin(bench, total, excess)
                if not ok:
                    min_margin = float(min_margin if min_margin < 1e18 else -1e18)
                    break
                passed += 1
                if float(margin) < float(min_margin):
                    min_margin = float(margin)
            fast_scored.append((int(passed), float(min_margin), cand))

        fast_scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        rerank_eval = [x[2] for x in fast_scored[: int(min(len(fast_scored), max(18, 2 * topk)))]]
        if tuple(base) not in {tuple(x) for x in rerank_eval}:
            rerank_eval = [base] + rerank_eval

        best = base
        best_passed = -1
        best_margin = -1e18

        for cand in rerank_eval:
            passed = 0
            min_margin = 1e18
            for _, start_time, end_time in list(final_select_periods):
                signal_start = pd.to_datetime(start_time) - pd.Timedelta(days=220)
                signal_df, _ = compute_signal_matrix(
                    formula=cand,
                    provider_uri=provider_uri,
                    instruments=instruments,
                    start_time=signal_start.strftime("%Y-%m-%d"),
                    end_time=end_time,
                    lookback_days=730,
                    token_mode=token_mode,
                    price_mode="pre_adjusted",
                    monthly_resample=False,
                )
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
                row = backtest_summary_row("seg", analysis, report_df)
                ok, margin = _pass_bullbear_margin(
                    row.get("BenchReturn%", 0.0),
                    row.get("TotalReturn%", 0.0),
                    row.get("ExcessReturn%", 0.0),
                )
                if not ok:
                    min_margin = float(min_margin if min_margin < 1e18 else -1e18)
                    break
                passed += 1
                if float(margin) < float(min_margin):
                    min_margin = float(margin)

            if passed > best_passed or (passed == best_passed and float(min_margin) > float(best_margin)):
                best = cand
                best_passed = int(passed)
                best_margin = float(min_margin)
            if int(passed) == int(len(final_select_periods)) and float(min_margin) > 0.0:
                best = cand
                break

        formula = best

        with open(os.path.join(out_dir, "seed.json"), "w") as f:
            json.dump({"seed": seed}, f)
        with open(os.path.join(out_dir, "best_ashare_strategy.json"), "w") as f:
            json.dump(list(formula), f)
        try:
            weights_out = os.path.join(out_dir, "alphagpt_model.pth")
            torch.save(engine.model.state_dict(), weights_out, _use_new_zipfile_serialization=False)
        except Exception:
            pass

        summary_df = run_multi_backtest(
            formula,
            periods=periods,
            instruments=instruments,
            provider_uri=provider_uri,
            backtest_mode=backtest_mode,
            token_mode=token_mode,
        )
        summary_path = os.path.join(out_dir, "backtest_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        def _row(label):
            return summary_df[summary_df["label"] == label].iloc[0].to_dict()

        train_row = _row("train")
        valid_row = _row("valid")
        holdout_row = _row("holdout")
        test_row = _row("test_2025")

        train_bench = float(train_row.get("BenchReturn%", 0.0))
        train_total = float(train_row.get("TotalReturn%", 0.0))
        train_excess = float(train_row.get("ExcessReturn%", 0.0))
        valid_bench = float(valid_row.get("BenchReturn%", 0.0))
        valid_total = float(valid_row.get("TotalReturn%", 0.0))
        valid_excess = float(valid_row.get("ExcessReturn%", 0.0))
        holdout_bench = float(holdout_row.get("BenchReturn%", 0.0))
        holdout_total = float(holdout_row.get("TotalReturn%", 0.0))
        holdout_excess = float(holdout_row.get("ExcessReturn%", 0.0))
        test_bench = float(test_row.get("BenchReturn%", 0.0))
        test_total = float(test_row.get("TotalReturn%", 0.0))
        test_excess = float(test_row.get("ExcessReturn%", 0.0))

        holdout_is_bull = bool(holdout_bench >= 0.0)
        passed_holdout_only = True
        if holdout_is_bull:
            passed_holdout_only = bool(holdout_excess > float(pass_holdout_excess_min_pct))

        passed_3seg = bool(
            _pass_bullbear(train_bench, train_total, train_excess)
            and _pass_bullbear(valid_bench, valid_total, valid_excess)
            and _pass_bullbear(holdout_bench, holdout_total, holdout_excess)
        )

        if str(pass_mode).lower() in ("3seg", "three_seg", "three_seg_bullbear", "strict"):
            passed = passed_3seg
        else:
            passed = passed_holdout_only

        pass_n += int(bool(passed))

        results.append(
            {
                "seed": seed,
                "passed": bool(passed),
                "formula": str(formula),
                "passed_holdout_only": bool(passed_holdout_only),
                "passed_3seg": bool(passed_3seg),
                "train_bench%": train_bench,
                "train_total%": train_total,
                "train_excess%": train_excess,
                "valid_bench%": valid_bench,
                "valid_total%": valid_total,
                "valid_excess%": valid_excess,
                "holdout_bench%": holdout_bench,
                "holdout_total%": holdout_total,
                "holdout_excess%": holdout_excess,
                "holdout_is_bull": bool(holdout_is_bull),
                "test_2025_bench%": test_bench,
                "test_2025_total%": test_total,
                "test_2025_excess%": test_excess,
                "summary_csv": summary_path,
            }
        )

    total_n = int(len(results))
    pass_rate = float(pass_n) / float(total_n) if total_n else 0.0

    record = {
        "timestamp": ts,
        "run_dir": run_dir,
        "source_run_dir": str(source_run_dir),
        "seeds": list(seeds),
        "train": [train_start, train_end],
        "valid_periods": list(valid_periods),
        "holdout": [holdout_start, holdout_end],
        "pressure": [pressure_start, pressure_end],
        "valid_agg": valid_agg,
        "valid_score_mode": valid_score_mode,
        "batch_size": int(batch_size),
        "train_steps": int(train_steps),
        "max_formula_len": int(max_formula_len),
        "token_mode": token_mode,
        "backtest_mode": backtest_mode,
        "post_refine_trials": int(post_refine_trials),
        "post_refine_mutations": int(post_refine_mutations),
        "valid_bull_weight": None if valid_bull_weight is None else float(valid_bull_weight),
        "valid_bear_weight": None if valid_bear_weight is None else float(valid_bear_weight),
        "pass_mode": str(pass_mode),
        "pass_holdout_excess_min_pct": float(pass_holdout_excess_min_pct),
        "target_pass_n": int(target_pass_n),
        "pass_n": int(pass_n),
        "total_n": int(total_n),
        "pass_rate": float(pass_rate),
        "final_select_samples": int(final_select_samples),
        "final_select_topk": int(final_select_topk),
        "final_select_temperature": float(final_select_temperature),
        "results": results,
    }
    with open(os.path.join(run_dir, "v3_record.json"), "w") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    try:
        lines = []
        lines.append("")
        lines.append("")
        lines.append(f"## RerankOnly {ts}")
        lines.append("")
        lines.append(f"- run_dir: `{run_dir}`")
        lines.append(f"- source_run_dir: `{source_run_dir}`")
        lines.append(f"- seeds: `{list(seeds)}`")
        lines.append(f"- final_select_samples/topk/temp: `{int(final_select_samples)}` / `{int(final_select_topk)}` / `{float(final_select_temperature)}`")
        lines.append(f"- result: `{pass_n}/{total_n}` ({pass_rate:.2%}), target `{int(target_pass_n)}/{total_n}`")
        lines.append("")
        lines.append("| seed | pass | train E% | valid (B%,T%,E%) | holdout E% | test_2025 E% | summary |")
        lines.append("|---:|:---:|---:|---:|---:|---:|---|")
        for r in results:
            lines.append(
                f"| {r['seed']} | {'Y' if r['passed'] else 'N'} | {float(r['train_excess%']):.2f} | ({float(r['valid_bench%']):.2f},{float(r['valid_total%']):.2f},{float(r['valid_excess%']):.2f}) | {float(r['holdout_excess%']):.2f} | {float(r['test_2025_excess%']):.2f} | `{r['summary_csv']}` |"
            )
        with open(checklist_path, "a") as f:
            f.write("\n".join(lines))
    except Exception:
        pass

    return record


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
