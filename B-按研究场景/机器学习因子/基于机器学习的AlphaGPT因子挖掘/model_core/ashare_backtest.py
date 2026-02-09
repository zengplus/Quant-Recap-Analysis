""" 
 File: ashare_backtest.py
 Date: 2026-01-17
 Description: A股回测评测模块。定义了 AShareBacktest 类，实现了基于 A 股交易规则（印花税、佣金）的因子评测逻辑，包含 Sortino 比率计算和分段验证机制。
 """ 
import torch
import pandas as pd

class AShareBacktest:
    def __init__(self, rebalance_freq="M"):
        # A股费率配置
        self.commission = 0.0003 # 万三佣金
        self.tax = 0.001        # 千一印花税 (仅卖出收取)
        self.min_cost = 5.0     # 最低消费 5 元 (简化起见，模型训练中通常忽略，仅在实盘考虑)
        self.rebalance_freq = rebalance_freq
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
        for i in range(len(month_keys) - 1):
            if month_keys[i] != month_keys[i + 1]:
                idx.append(i)
        if len(month_keys) > 1:
            idx.append(len(month_keys) - 1)
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
        # --- 1. 因子预处理 (Cross-Sectional Norm) ---
        # 使用截面数据归一化，消除 Beta，聚焦 Alpha
        # FIX: 改为 dim=0 (Assets) 截面归一化，原 dim=1 (Time) 为全局未来函数
        mean = factors.mean(dim=0, keepdim=True)
        std = factors.std(dim=0, keepdim=True) + 1e-9
        factors_norm = (factors - mean) / std
        
        # --- 2. 生成信号与持仓 ---
        # 使用 sigmoid 将标准化后的因子映射到 (0, 1)
        # 相比硬阈值 >0.7，保留了信号强弱信息，但为了模拟实际交易，
        # 我们仍需截断或加权。这里采用软截断：
        # 信号 < 0.5 做空(A股不可做空则为0)，> 0.5 做多
        # 在 A 股 Long-Only 场景下，我们关注 ranking。
        # 这里沿用 Sigmoid 但增加缩放因子，使其更平滑
        signal = torch.sigmoid(factors_norm * 2.0) 
        
        if self.rebalance_freq == "M" and dates is not None:
            rebalance_idx = self._get_rebalance_idx(dates, device=factors.device)
            position = torch.empty_like(signal)
            pos_at_rebalance = torch.clamp((signal.index_select(1, rebalance_idx) - 0.6) / 0.4, min=0.0, max=1.0)
            idx_list = rebalance_idx.tolist()
            t_len = signal.shape[1]
            for k, start in enumerate(idx_list):
                end = idx_list[k + 1] if (k + 1) < len(idx_list) else t_len
                position[:, start:end] = pos_at_rebalance[:, k].unsqueeze(1)
        else:
            position = torch.clamp((signal - 0.6) / 0.4, min=0.0, max=1.0)
        
        # --- 3. 计算换手与成本 ---
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        
        buy_vol = torch.clamp(position - prev_pos, min=0)
        sell_vol = torch.clamp(prev_pos - position, min=0)
        
        cost_buy = buy_vol * self.commission
        cost_sell = sell_vol * (self.commission + self.tax)
        total_cost = cost_buy + cost_sell
        
        # --- 4. 计算净收益 ---
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - total_cost
        
        # --- 5. 高级指标计算 (参考 1.py) ---
        # 5.1 累计收益与年化收益
        cum_ret = net_pnl.sum(dim=1)
        
        # 5.2 Sortino Ratio (比 Sharpe 更优秀)
        # 仅惩罚下行波动
        annual_factor = 15.87 # sqrt(252)
        mu = net_pnl.mean(dim=1)
        # 下行标准差
        down_returns = torch.clamp(net_pnl, max=0.0)
        down_std = torch.sqrt((down_returns ** 2).mean(dim=1)) + 1e-9
        sortino = (mu / down_std) * annual_factor
        
        # 5.3 分段验证 (防止过拟合)
        # 将时间轴分为 3 段，要求因子在至少 2 段中表现良好
        seq_len = net_pnl.shape[1]
        seg_len = seq_len // 3
        
        score_penalty = torch.zeros_like(cum_ret)
        
        if seq_len > 60:
            seg1 = net_pnl[:, :seg_len]
            seg2 = net_pnl[:, seg_len:2*seg_len]
            seg3 = net_pnl[:, 2*seg_len:]
            
            valid_segs = 0
            # 检查每段的夏普/均值是否为正
            valid_segs += (seg1.mean(dim=1) > 0).float()
            valid_segs += (seg2.mean(dim=1) > 0).float()
            valid_segs += (seg3.mean(dim=1) > 0).float()
            
            # 如果有效段数 < 2，给予重罚
            score_penalty = torch.where(valid_segs < 2, torch.tensor(-50.0, device=factors.device), torch.tensor(0.0, device=factors.device))

        # --- 6. 综合评分 ---
        # 基础分：Sortino (注重风险调整后收益) + 累计收益
        score = sortino * 2.0 + cum_ret * 10.0
        
        # 加上分段惩罚
        score = score + score_penalty
        
        # 活跃度检查 (避免几乎不交易的僵尸策略)
        activity = (position > 0.01).float().sum(dim=1)
        score = torch.where(activity < 10, torch.tensor(-100.0, device=score.device), score)
        
        return torch.median(score), cum_ret.mean().item()