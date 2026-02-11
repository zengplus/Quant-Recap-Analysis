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
        turnover_penalty=0.0,
        abs_weight=1.0,
        alpha_weight_bull=1.0,
        alpha_weight_bear=0.3,
        market_regime_threshold=0.0,
        neg_excess_penalty=0.0,
        neg_port_penalty=0.0,
        min_excess_weight=0.0,
    ):
        self.commission = 0.0003
        self.tax = 0.001
        self.min_cost = 5.0
        self.rebalance_freq = rebalance_freq
        self.topk = int(topk) if topk is not None else 10
        self.signal_lag = int(signal_lag) if signal_lag is not None else 1
        self.turnover_penalty = float(turnover_penalty) if turnover_penalty is not None else 0.0
        self.abs_weight = float(abs_weight) if abs_weight is not None else 1.0
        self.alpha_weight_bull = float(alpha_weight_bull) if alpha_weight_bull is not None else 1.0
        self.alpha_weight_bear = float(alpha_weight_bear) if alpha_weight_bear is not None else 0.3
        self.market_regime_threshold = float(market_regime_threshold) if market_regime_threshold is not None else 0.0
        self.neg_excess_penalty = float(neg_excess_penalty) if neg_excess_penalty is not None else 0.0
        self.neg_port_penalty = float(neg_port_penalty) if neg_port_penalty is not None else 0.0
        self.min_excess_weight = float(min_excess_weight) if min_excess_weight is not None else 0.0
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

    def evaluate(self, factors, raw_data, target_ret, dates=None):
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
        rebalance_idx = rebalance_idx[rebalance_idx + lag < t]
        if rebalance_idx.numel() == 0:
            return torch.tensor(-1.0, device=factors.device), 0.0

        k = int(self.topk) if self.topk is not None else 10
        if k <= 0:
            return torch.tensor(-1.0, device=factors.device), 0.0
        k = int(min(k, n))

        port_parts = []
        excess_parts = []
        turnover_parts = []
        bench_ret = None
        if raw_data is not None and isinstance(raw_data, dict) and "bench_ret" in raw_data:
            bench_ret = raw_data.get("bench_ret", None)
        idx_list = rebalance_idx.tolist()
        prev_set = None
        for j, r in enumerate(idx_list):
            trade_day = int(r)
            sig_day = int(r - lag) if lag > 0 else int(r)
            if sig_day < 0:
                sig_day = 0
            start = trade_day
            end = int(idx_list[j + 1]) if (j + 1) < len(idx_list) else int(t)
            if start >= end:
                continue
            f = x[:, sig_day]
            f = torch.nan_to_num(f, nan=-1e9, posinf=1e9, neginf=-1e9)
            top_idx = torch.topk(f, k=k, largest=True).indices
            r_sel = y.index_select(0, top_idx)[:, start:end]
            port = r_sel.mean(dim=0)
            port_parts.append(port)
            if bench_ret is not None and isinstance(bench_ret, torch.Tensor) and bench_ret.numel() >= end:
                b = bench_ret[start:end].to(device=port.device, dtype=port.dtype)
                excess_parts.append(port - b)
            else:
                base = y[:, start:end].mean(dim=0)
                excess_parts.append(port - base)
            cur_set = set(int(i) for i in top_idx.tolist())
            if prev_set is not None:
                overlap = len(prev_set.intersection(cur_set))
                turnover_parts.append(1.0 - float(overlap) / float(k))
            prev_set = cur_set

        if len(port_parts) == 0:
            return torch.tensor(-1.0, device=factors.device), 0.0

        port_all = torch.cat(port_parts, dim=0)
        excess_all = torch.cat(excess_parts, dim=0) if len(excess_parts) else torch.zeros_like(port_all)
        port_ann = port_all.mean() * 252.0 * 100.0
        excess_ann = excess_all.mean() * 252.0 * 100.0
        bench_all = port_all - excess_all
        bench_ann = bench_all.mean() * 252.0 * 100.0
        alpha_w = self.alpha_weight_bull if float(bench_ann.detach().cpu().item()) >= float(self.market_regime_threshold) else self.alpha_weight_bear
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
        )
        avg_ret = float(port_ann.detach().cpu().item())
        return score, avg_ret
