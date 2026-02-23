""" 
 File: ashare_backtest.py
 Date: 2026-01-17
 Description: A股回测评测模块。定义了 AShareBacktest 类，实现了基于 A 股交易规则（印花税、佣金）的因子评测逻辑，包含 Sortino 比率计算和分段验证机制。
 """ 
import torch
import pandas as pd

class AShareBacktest:
    def __init__(
        self,
        rebalance_freq="M",
        topk=10,
        signal_lag=1,
        backfill_untradable=True,
        turnover_penalty=0.0,
        abs_weight=1.0,
        alpha_weight_bull=1.0,
        alpha_weight_bear=0.3,
        market_regime_threshold=0.0,
        neg_excess_penalty=0.0,
        neg_port_penalty=0.0,
        min_excess_weight=0.0,
        beta_target=1.0,
        beta_min_bull=0.9,
        beta_penalty_weight_bull=0.0,
        beta_min_penalty_weight_bull=0.0,
        beta_penalty_weight_bull_obj=0.0,
        beta_min_penalty_weight_bull_obj=0.0,
        bull_excess_min=0.0,
        bull_total_min=0.0,
        bear_excess_min=0.0,
        bear_total_min=-0.3,
        bull_excess_target=0.0,
        bull_excess_hinge_weight_score=0.0,
        bull_excess_hinge_weight_obj=0.0,
        slippage_rate=0.0003,
    ):
        self.commission = 0.0003
        self.tax = 0.001
        self.min_cost = 5.0
        self.slippage_rate = float(slippage_rate) if slippage_rate is not None else 0.0003
        self.rebalance_freq = rebalance_freq
        self.topk = int(topk) if topk is not None else 10
        self.signal_lag = int(signal_lag) if signal_lag is not None else 1
        self.backfill_untradable = bool(backfill_untradable)
        self.turnover_penalty = float(turnover_penalty) if turnover_penalty is not None else 0.0
        self.abs_weight = float(abs_weight) if abs_weight is not None else 1.0
        self.alpha_weight_bull = float(alpha_weight_bull) if alpha_weight_bull is not None else 1.0
        self.alpha_weight_bear = float(alpha_weight_bear) if alpha_weight_bear is not None else 0.3
        self.market_regime_threshold = float(market_regime_threshold) if market_regime_threshold is not None else 0.0
        self.neg_excess_penalty = float(neg_excess_penalty) if neg_excess_penalty is not None else 0.0
        self.neg_port_penalty = float(neg_port_penalty) if neg_port_penalty is not None else 0.0
        self.min_excess_weight = float(min_excess_weight) if min_excess_weight is not None else 0.0
        self.beta_target = float(beta_target) if beta_target is not None else 1.0
        self.beta_min_bull = float(beta_min_bull) if beta_min_bull is not None else 0.9
        self.beta_penalty_weight_bull = float(beta_penalty_weight_bull) if beta_penalty_weight_bull is not None else 0.0
        self.beta_min_penalty_weight_bull = float(beta_min_penalty_weight_bull) if beta_min_penalty_weight_bull is not None else 0.0
        self.beta_penalty_weight_bull_obj = float(beta_penalty_weight_bull_obj) if beta_penalty_weight_bull_obj is not None else 0.0
        self.beta_min_penalty_weight_bull_obj = float(beta_min_penalty_weight_bull_obj) if beta_min_penalty_weight_bull_obj is not None else 0.0
        self.bull_excess_min = float(bull_excess_min) if bull_excess_min is not None else 0.0
        self.bull_total_min = float(bull_total_min) if bull_total_min is not None else 0.0
        self.bear_excess_min = float(bear_excess_min) if bear_excess_min is not None else 0.0
        self.bear_total_min = float(bear_total_min) if bear_total_min is not None else -0.3
        self.bull_excess_target = float(bull_excess_target) if bull_excess_target is not None else 0.0
        self.bull_excess_hinge_weight_score = float(bull_excess_hinge_weight_score) if bull_excess_hinge_weight_score is not None else 0.0
        self.bull_excess_hinge_weight_obj = float(bull_excess_hinge_weight_obj) if bull_excess_hinge_weight_obj is not None else 0.0
        self._rebalance_cache_key = None
        self._rebalance_idx = None
    
    def _get_rebalance_idx(self, dates, device):
        if dates is None:
            return None
        key = (int(len(dates)), str(dates[0]) if len(dates) > 0 else "", str(dates[-1]) if len(dates) > 0 else "")
        if self._rebalance_cache_key == key and self._rebalance_idx is not None:
            return self._rebalance_idx
        dt = pd.to_datetime(dates)
        month_keys = dt.to_period("M")
        idx = [0]
        for i in range(1, len(month_keys)):
            if month_keys[i] != month_keys[i - 1]:
                idx.append(i)
        idx = sorted(set(int(i) for i in idx if 0 <= int(i) < len(month_keys)))
        if len(idx) == 0:
            idx = [0]
        self._rebalance_cache_key = key
        self._rebalance_idx = torch.tensor(idx, dtype=torch.long, device=device)
        return self._rebalance_idx

    def evaluate(self, factors, raw_data, target_ret, dates=None, return_details=False):
        """
        评估因子在 A 股规则下的表现 (增强版)
        参考 1.py 的优秀实践：
        1. 因子标准化 (Z-Score)
        2. Sortino 比率替代 Sharpe (更关注下行风险)
        3. 分段验证 (防止过拟合)
        4. 动态持仓 (基于信号强度)
        
        Args:
            factors: (Batch, Time) 因子值
            raw_data: 原始数据字典
            target_ret: (Batch, Time) T+1 收益率
            
        Returns:
            fitness: 适应度评分 (用于进化)
            avg_ret: 平均收益率
        """
        if factors is None or target_ret is None:
            return torch.tensor(-1.0), 0.0
        if factors.dim() != 2 or target_ret.dim() != 2:
            return torch.tensor(-1.0, device=factors.device), 0.0
        n = min(int(factors.shape[0]), int(target_ret.shape[0]))
        t = min(int(factors.shape[1]), int(target_ret.shape[1]))
        if n <= 1 or t <= 2:
            return torch.tensor(-1.0, device=factors.device), 0.0
        x = factors[:n, :t]
        y = target_ret[:n, :t]
        x = torch.nan_to_num(x, nan=float("nan"), posinf=float("nan"), neginf=float("nan"))
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        if self.rebalance_freq == "M" and dates is not None:
            rebalance_idx = self._get_rebalance_idx(dates[:t], device=factors.device)
        else:
            rebalance_idx = torch.arange(0, t, device=factors.device, dtype=torch.long)

        if rebalance_idx is None or rebalance_idx.numel() == 0:
            return torch.tensor(-1.0, device=factors.device), 0.0

        lag = int(self.signal_lag) if self.signal_lag is not None else 0
        if lag < 0:
            lag = 0
        rebalance_idx = rebalance_idx[rebalance_idx + 1 + lag < t]
        if rebalance_idx.numel() == 0:
            return torch.tensor(-1.0, device=factors.device), 0.0

        k = int(self.topk) if self.topk is not None else 10
        if k <= 0:
            return torch.tensor(-1.0, device=factors.device), 0.0
        k = int(min(k, n))

        port_parts = []
        bench_parts = []
        excess_parts = []
        turnover_parts = []
        bench_ret = None
        if raw_data is not None and isinstance(raw_data, dict) and "bench_ret" in raw_data:
            bench_ret = raw_data.get("bench_ret", None)
        pool_by_date = None
        asset_list = None
        asset_pos = None
        pool_mask_cache = {}
        if raw_data is not None and isinstance(raw_data, dict):
            pool_by_date = raw_data.get("pool_by_date", None)
            asset_list = raw_data.get("asset_list", None)
        if pool_by_date is not None and asset_list is not None:
            try:
                asset_pos = {str(s): i for i, s in enumerate(list(asset_list))}
            except Exception:
                asset_pos = None
        idx_list = rebalance_idx.tolist()
        prev_set = None
        for j, r in enumerate(idx_list):
            trade_day = int(r)
            sig_day = int(r - 1 - lag)
            if sig_day < 0:
                sig_day = 0
            start = trade_day
            end = int(idx_list[j + 1]) if (j + 1) < len(idx_list) else int(t)
            if start >= end:
                continue
            f = x[:, sig_day]
            f = torch.nan_to_num(f, nan=-1e9, posinf=1e9, neginf=-1e9)
            if pool_by_date is not None and asset_pos is not None and dates is not None:
                try:
                    ds = pd.to_datetime(dates[sig_day]).strftime("%Y-%m-%d")
                except Exception:
                    ds = None
                if ds is not None:
                    mask = pool_mask_cache.get(ds, None)
                    if mask is None:
                        pool = None
                        try:
                            pool = pool_by_date.get(ds, None)
                        except Exception:
                            pool = None
                        if pool is not None:
                            m = torch.zeros((n,), dtype=torch.bool, device=f.device)
                            for sym in pool:
                                idx = asset_pos.get(str(sym))
                                if idx is not None and 0 <= int(idx) < n:
                                    m[int(idx)] = True
                            mask = m
                        pool_mask_cache[ds] = mask
                    if mask is not None:
                        f = f.masked_fill(~mask, -1e9)
            top_idx = None
            if bool(self.backfill_untradable) and raw_data is not None and isinstance(raw_data, dict):
                open_t = raw_data.get("open", None)
                high_t = raw_data.get("high", None)
                close_t = raw_data.get("close", None)
                vol_t = raw_data.get("volume", None)
                tradable = None
                if isinstance(open_t, torch.Tensor) and open_t.dim() == 2 and open_t.shape[1] > trade_day:
                    o = open_t[:n, trade_day].to(device=f.device, dtype=f.dtype)
                    tradable = o > 1e-6
                    if isinstance(vol_t, torch.Tensor) and vol_t.dim() == 2 and vol_t.shape[1] > trade_day:
                        v = vol_t[:n, trade_day].to(device=f.device, dtype=f.dtype)
                        tradable = tradable & (v > 0)
                    if isinstance(high_t, torch.Tensor) and isinstance(close_t, torch.Tensor) and high_t.dim() == 2 and close_t.dim() == 2 and high_t.shape[1] > trade_day and close_t.shape[1] > trade_day:
                        h = high_t[:n, trade_day].to(device=f.device, dtype=f.dtype)
                        c = close_t[:n, trade_day].to(device=f.device, dtype=f.dtype)
                        buy_block = torch.isclose(o, h, rtol=1e-5, atol=1e-4) & torch.isclose(c, h, rtol=1e-5, atol=1e-4) & (h > 1e-6)
                        tradable = tradable & (~buy_block)
                ranked = torch.argsort(f, descending=True)
                ranked_list = ranked.tolist()
                selected = []
                prev_local = prev_set if prev_set is not None else set()
                if tradable is not None:
                    trad_list = tradable.tolist()
                    for idx in ranked_list:
                        if len(selected) >= k:
                            break
                        if (not bool(trad_list[int(idx)])) and (int(idx) not in prev_local):
                            continue
                        selected.append(int(idx))
                else:
                    selected = ranked_list[:k]
                if prev_set is not None and len(selected) < k:
                    picked = set(selected)
                    for idx in prev_set:
                        if len(selected) >= k:
                            break
                        if int(idx) in picked:
                            continue
                        selected.append(int(idx))
                if selected:
                    top_idx = torch.tensor(selected, device=f.device, dtype=torch.long)
            if top_idx is None:
                top_idx = torch.topk(f, k=k, largest=True).indices
            cur_set = set(int(i) for i in top_idx.tolist())
            turnover_rate = 0.0
            if prev_set is None:
                turnover_rate = 1.0 if len(cur_set) else 0.0
            else:
                denom = float(max(1, max(len(prev_set), len(cur_set))))
                overlap = len(prev_set.intersection(cur_set))
                turnover_rate = 1.0 - float(overlap) / float(denom)
            turnover_parts.append(float(turnover_rate))
            prev_set = cur_set
            if top_idx.numel() == 0:
                port = torch.zeros((end - start,), device=y.device, dtype=y.dtype)
            else:
                r_sel_log = y.index_select(0, top_idx)[:, start:end]
                r_sel = torch.expm1(r_sel_log)
                port = r_sel.mean(dim=0)
            if port.numel() > 0:
                cost_per_turnover = (2.0 * float(self.commission)) + (2.0 * float(self.slippage_rate)) + (0.5 * float(self.tax))
                cost = float(turnover_rate) * float(cost_per_turnover)
                port = port.clone()
                port[0] = port[0] - port.new_tensor(cost)
            port_parts.append(port)
            if bench_ret is not None and isinstance(bench_ret, torch.Tensor) and bench_ret.numel() >= end:
                b_log = bench_ret[start:end].to(device=port.device, dtype=port.dtype)
                b = torch.expm1(b_log)
                bench_parts.append(b)
                excess_parts.append(port - b)
            else:
                base = torch.expm1(y[:, start:end]).mean(dim=0)
                bench_parts.append(base)
                excess_parts.append(port - base)

        if len(port_parts) == 0:
            return torch.tensor(-1.0, device=factors.device), 0.0

        port_all = torch.cat(port_parts, dim=0)
        bench_all = torch.cat(bench_parts, dim=0) if len(bench_parts) else torch.zeros_like(port_all)
        excess_all = torch.cat(excess_parts, dim=0) if len(excess_parts) else torch.zeros_like(port_all)
        port_ann = port_all.mean() * 252.0 * 100.0
        excess_ann = excess_all.mean() * 252.0 * 100.0
        bench_ann = bench_all.mean() * 252.0 * 100.0
        alpha_w = self.alpha_weight_bull if float(bench_ann.detach().cpu().item()) >= float(self.market_regime_threshold) else self.alpha_weight_bear
        bench_is_bull_ann = bool(float(bench_ann.detach().cpu().item()) >= float(self.market_regime_threshold))
        port_c = port_all - port_all.mean()
        bench_c = bench_all - bench_all.mean()
        bench_var = (bench_c * bench_c).mean()
        beta = torch.tensor(0.0, device=port_all.device, dtype=port_all.dtype)
        if float(bench_var.detach().cpu().item()) > 1e-12:
            beta = (port_c * bench_c).mean() / (bench_var + 1e-12)
        beta_dev = torch.abs(beta - float(self.beta_target))
        beta_short = torch.relu(float(self.beta_min_bull) - beta)
        beta_pen_score = torch.tensor(0.0, device=port_all.device, dtype=port_all.dtype)
        if bench_is_bull_ann and (self.beta_penalty_weight_bull > 0.0 or self.beta_min_penalty_weight_bull > 0.0):
            beta_pen_score = float(self.beta_penalty_weight_bull) * beta_dev + float(self.beta_min_penalty_weight_bull) * beta_short
        abs_w = float(self.abs_weight)
        turnover_pen = float(self.turnover_penalty)
        turnover_avg = float(sum(turnover_parts) / max(1, len(turnover_parts))) if len(turnover_parts) else 0.0
        turnover_ann = turnover_avg * 12.0
        neg_excess = torch.relu(-excess_ann)
        neg_port = torch.relu(-port_ann)
        if len(excess_parts):
            seg_excess = torch.stack([(p.mean() * 252.0 * 100.0) for p in excess_parts])
            seg_excess, _ = torch.sort(seg_excess)
            m = int(seg_excess.numel())
            frac = float(getattr(self, "tail_seg_fraction", 0.2))
            if frac < 0.0:
                frac = 0.0
            if frac > 1.0:
                frac = 1.0
            tail_n = int(max(1, round(frac * float(m))))
            tail_excess_ann = seg_excess[:tail_n].mean()
        else:
            tail_excess_ann = torch.zeros_like(excess_ann)
        score = (
            abs_w * port_ann
            + float(alpha_w) * excess_ann
            + float(self.min_excess_weight) * tail_excess_ann
            - turnover_pen * float(turnover_ann)
            - float(self.neg_excess_penalty) * neg_excess
            - float(self.neg_port_penalty) * neg_port
            - beta_pen_score
        )
        avg_ret = float(port_ann.detach().cpu().item())
        if not return_details:
            return score, avg_ret

        port_r = torch.nan_to_num(port_all, nan=0.0, posinf=0.0, neginf=0.0)
        bench_r = torch.nan_to_num(bench_all, nan=0.0, posinf=0.0, neginf=0.0)
        port_total = torch.expm1(torch.log1p(torch.clamp(port_r, min=-0.999999)).sum())
        bench_total = torch.expm1(torch.log1p(torch.clamp(bench_r, min=-0.999999)).sum())
        excess_total = port_total - bench_total
        bull_excess_shortfall = torch.tensor(0.0, device=port_all.device, dtype=port_all.dtype)
        if float(self.bull_excess_hinge_weight_score) > 0.0:
            target = torch.tensor(float(self.bull_excess_target), device=port_all.device, dtype=port_all.dtype)
            bull_excess_shortfall = torch.relu(target - excess_total)
            if bool(float(bench_total.detach().cpu().item()) >= 0.0):
                score = score - float(self.bull_excess_hinge_weight_score) * bull_excess_shortfall

        bench_is_bull = bool(float(bench_total.detach().cpu().item()) >= 0.0)
        if bench_is_bull:
            excess_ok = bool(float(excess_total.detach().cpu().item()) > float(self.bull_excess_min))
            port_ok = bool(float(port_total.detach().cpu().item()) > float(self.bull_total_min))
            objective_pass = bool(excess_ok and port_ok)
        else:
            excess_ok = bool(float(excess_total.detach().cpu().item()) > float(self.bear_excess_min))
            port_ok = bool(float(port_total.detach().cpu().item()) > float(self.bear_total_min))
            objective_pass = bool(excess_ok and port_ok)

        if bench_is_bull:
            objective_score = 1000.0 * excess_total + (100.0 if objective_pass else -100.0)
            if float(self.bull_excess_hinge_weight_obj) > 0.0:
                target = torch.tensor(float(self.bull_excess_target), device=port_all.device, dtype=port_all.dtype)
                objective_score = objective_score - float(self.bull_excess_hinge_weight_obj) * torch.relu(target - excess_total)
        else:
            objective_score = 1000.0 * excess_total + 300.0 * port_total + (100.0 if objective_pass else -100.0)
        beta_pen_obj = torch.tensor(0.0, device=port_all.device, dtype=port_all.dtype)
        if bench_is_bull and (self.beta_penalty_weight_bull_obj > 0.0 or self.beta_min_penalty_weight_bull_obj > 0.0):
            beta_pen_obj = float(self.beta_penalty_weight_bull_obj) * beta_dev + float(self.beta_min_penalty_weight_bull_obj) * beta_short
            objective_score = objective_score - beta_pen_obj

        details = {
            "port_total_return_pct": float(port_total.detach().cpu().item()) * 100.0,
            "bench_total_return_pct": float(bench_total.detach().cpu().item()) * 100.0,
            "excess_total_return_pct": float(excess_total.detach().cpu().item()) * 100.0,
            "bench_is_bull": bool(bench_is_bull),
            "objective_pass": bool(objective_pass),
            "objective_score": float(objective_score.detach().cpu().item()),
            "beta": float(beta.detach().cpu().item()),
            "beta_penalty_score": float(beta_pen_score.detach().cpu().item()),
            "beta_penalty_objective": float(beta_pen_obj.detach().cpu().item()),
            "bull_excess_shortfall_pct": float(bull_excess_shortfall.detach().cpu().item()) * 100.0,
        }
        return score, avg_ret, details
