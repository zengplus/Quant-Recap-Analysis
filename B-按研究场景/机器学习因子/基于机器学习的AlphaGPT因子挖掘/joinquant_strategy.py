import numpy as np
import pandas as pd
import torch
import json
from jqdata import *
from datetime import timedelta

# ----------------------
# 1. 全局配置 (Global Configuration)
# ----------------------
# 将配置定义为模块级常量，避免直接定义 class g 导致与聚宽框架的全局 g 对象冲突
_BENCHMARK = '000300.XSHG'
_MAX_HOLD_STOCKS = 10
_TOP_K = 10
_TRADE_TIME = 'open'
_HOLDING_PERIOD = 20
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_COMMISSION_RATE = 0.0003
_STAMP_TAX_RATE = 0.001
_MIN_COMMISSION = 5.0
_SLIPPAGE_RATE = 0.0003

# 新训练的公式 (three_retrain_fast_20260208_155524/seed_2026020813)
_FORMULA_CANDIDATES = {
    "Pipeline-Generated": {
        "token_mode": "postfix",
        "tokens": [11, 17, 13, 12, 10, 13, 14, 12, 0, 13, 12, 17],
    },
}
_ACTIVE_FORMULA = "Pipeline-Generated"


def _normalize_token_mode(mode):
    m = str(mode or "postfix").strip().lower()
    return "prefix" if m == "prefix" else "postfix"

def _repair_postfix(tokens):
    if not tokens:
        return [0]
    stack_size = 0
    valid = []
    add_token = int(len(_FEATURES))
    for t in tokens:
        try:
            t = int(t)
        except Exception:
            continue
        if t < len(_FEATURES):
            stack_size += 1
            valid.append(t)
            continue
        arity = _OP_ARITY_MAP.get(t)
        if arity is None:
            continue
        if stack_size >= int(arity):
            stack_size -= (int(arity) - 1)
            valid.append(t)
    if stack_size == 0:
        valid.append(0)
        stack_size = 1
    while stack_size > 1:
        valid.append(add_token)
        stack_size -= 1
    return valid


def _repair_prefix(tokens):
    if not tokens:
        return [0]
    rev = list(reversed(tokens))
    repaired = _repair_postfix(rev)
    return list(reversed(repaired))



def _load_formula_config():
    cfg = dict((_FORMULA_CANDIDATES or {}).get(_ACTIVE_FORMULA) or {})
    tokens = cfg.get("tokens", [0])
    mode = _normalize_token_mode(cfg.get("token_mode"))
    source = f"candidates:{_ACTIVE_FORMULA}"
    try:
        txt = read_file("best_ashare_strategy.json")
        if txt:
            if isinstance(txt, (bytes, bytearray)):
                try:
                    txt = txt.decode("utf-8", errors="ignore")
                except Exception:
                    txt = str(txt)
            txt = str(txt).strip()
            if txt:
                obj = json.loads(txt)
                loaded = None
                if isinstance(obj, (list, tuple)):
                    loaded = list(obj)
                elif isinstance(obj, dict):
                    if isinstance(obj.get("tokens"), (list, tuple)):
                        loaded = list(obj.get("tokens"))
                if loaded is not None:
                    tokens = loaded
                    source = "file:best_ashare_strategy.json"
    except Exception:
        pass
    if not isinstance(tokens, (list, tuple)):
        tokens = [0]
    out = []
    for t in tokens:
        try:
            out.append(int(t))
        except Exception:
            continue
    if not out:
        out = [0]
    return out, mode, source

# ----------------------
# 2. 聚宽初始化函数
# ----------------------
def initialize(context):
    set_benchmark(_BENCHMARK)
    set_option('use_real_price', True)
    set_option("avoid_future_data", True)
    log.set_level('order', 'error')

    try:
        set_order_cost(
            OrderCost(
                open_tax=0.0,
                close_tax=float(_STAMP_TAX_RATE),
                open_commission=float(_COMMISSION_RATE),
                close_commission=float(_COMMISSION_RATE),
                min_commission=float(_MIN_COMMISSION),
            ),
            type="stock",
        )
        log.info(f"交易成本设置 commission={_COMMISSION_RATE} stamp_tax={_STAMP_TAX_RATE} min_commission={_MIN_COMMISSION}")
    except Exception as e:
        log.warn(f"交易成本设置失败: {e}")

    try:
        set_slippage(PriceRelatedSlippage(float(_SLIPPAGE_RATE)))
        log.info(f"滑点设置 PriceRelatedSlippage rate={_SLIPPAGE_RATE}")
    except Exception as e:
        try:
            set_slippage(FixedSlippage(0.0))
            log.warn(f"滑点设置回退 FixedSlippage(0.0)，原因: {e}")
        except Exception as e2:
            log.warn(f"滑点设置失败: {e2}")
    
    # === 初始化全局变量 g ===
    # 将模块常量赋值给 g 对象
    g.BENCHMARK = _BENCHMARK
    g.MAX_HOLD_STOCKS = _MAX_HOLD_STOCKS
    g.TOP_K = _TOP_K
    g.TRADE_TIME = _TRADE_TIME
    g.HOLDING_PERIOD = _HOLDING_PERIOD
    
    g.DEVICE = _DEVICE
    
    # 初始化算子和特征配置
    init_ops_and_features()
    
    g.FORMULA_CANDIDATES = _FORMULA_CANDIDATES
    g.ACTIVE_FORMULA = _ACTIVE_FORMULA

    tokens, mode, source = _load_formula_config()
    g.FORMULA_TOKENS = tokens
    g.TOKEN_MODE = mode
    g.FORMULA_SOURCE = source
    context.miner = StrategyMiner()
    context.trade_count = 0
    
    context.miner.set_formula(tokens, mode)
    log.info(f"公式: {context.miner.decode_expression(context.miner.best_formula_tokens)}")
    try:
        log.info(f"公式来源: {g.FORMULA_SOURCE} token_mode={g.TOKEN_MODE}")
        log.info(f"公式tokens(raw): {context.miner.raw_formula_tokens}")
        log.info(f"公式tokens(repaired): {context.miner.best_formula_tokens}")
    except Exception:
        pass
    
    # 尝试加载离线选股文件（可选）
    try:
        content = read_file('offline_targets.csv')
        g.offline_targets = {}
        if content:
            lines = content.strip().split('\n')
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 2:
                    d = parts[0].strip()
                    c = parts[1].strip()
                    if d not in g.offline_targets:
                        g.offline_targets[d] = []
                    g.offline_targets[d].append(c)
            log.info(f"成功加载选股文件，包含 {len(g.offline_targets)} 个交易日的数据")
    except:
        g.offline_targets = {}
        
    log.set_level('order', 'error')
    log.set_level('strategy', 'info')

    # ========== 明确在开盘时调仓 ==========
    run_monthly(rebalance, 1, time='open')  # 每月第一个交易日开盘时运行

    # ========== 交易时间配置 ==========
    g.trade_times = {
        'prepare': '9:10',
        'buy_main': '9:30',
        'buy_secondary': '10:00',
        'sell_check': '14:30',
        'sell_final': '14:50',
        'position_check': '15:05'
    }

# ----------------------
# 3. 算子与特征定义
# ----------------------
# 定义模块级全局变量，供 StrategyMiner 直接访问
_OPS_CONFIG = []
_FEATURES = []
_VOCAB = []
_OP_FUNC_MAP = {}
_OP_ARITY_MAP = {}


def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    """滞后算子"""
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

def _op_jump(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)

def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

def clean_torch_tensor(x):
    """清理张量"""
    if hasattr(torch, "isnan"):
        x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device), x)
    else:
        x = torch.where(x != x, torch.tensor(0.0, device=x.device), x)
    if hasattr(torch, "isinf"):
        x = torch.where(torch.isinf(x), torch.tensor(0.0, device=x.device), x)
    else:
        x = torch.where((x == float("inf")) | (x == -float("inf")), torch.tensor(0.0, device=x.device), x)
    x = torch.where(x == -float('inf'), torch.tensor(0.0, device=x.device), x)
    return x

def wrap_op(func):
    return lambda *args: torch.nan_to_num(func(*args), nan=0.0, posinf=0.0, neginf=0.0)

def init_ops_and_features():
    """初始化全局变量和 g 对象中的配置"""
    global _OPS_CONFIG, _FEATURES, _VOCAB, _OP_FUNC_MAP, _OP_ARITY_MAP
    
    _OPS_CONFIG = [
        ('ADD', wrap_op(lambda x, y: x + y), 2),
        ('SUB', wrap_op(lambda x, y: x - y), 2),
        ('MUL', wrap_op(lambda x, y: x * y), 2),
        ('DIV', wrap_op(lambda x, y: x / (y + 1e-6)), 2),
        ('NEG', wrap_op(lambda x: -x), 1),
        ('ABS', wrap_op(torch.abs), 1),
        ('SIGN', wrap_op(torch.sign), 1),
        ('GATE', wrap_op(_op_gate), 3),
        ('JUMP', wrap_op(_op_jump), 1),
        ('DECAY', wrap_op(_op_decay), 1),
        ('DELAY1', wrap_op(lambda x: _ts_delay(x, 1)), 1),
        ('MAX3', wrap_op(lambda x: torch.max(x, torch.max(_ts_delay(x, 1), _ts_delay(x, 2)))), 1),
    ]

    _FEATURES = [
        'RET',           # 收盘价收益率 (log return)
        'LIQ_SCORE',     # 流动性健康度
        'PRESSURE',      # 买卖压力
        'FOMO',          # 交易热度加速
        'DEV',           # 价格偏离度
        'LOG_VOL'        # 对数成交量
    ]

    _VOCAB = _FEATURES + [cfg[0] for cfg in _OPS_CONFIG]
    _OP_FUNC_MAP = {i + len(_FEATURES): cfg[1] for i, cfg in enumerate(_OPS_CONFIG)}
    _OP_ARITY_MAP = {i + len(_FEATURES): cfg[2] for i, cfg in enumerate(_OPS_CONFIG)}

# ----------------------
# 4. 辅助指标与特征工程
# ----------------------
class MemeIndicators:
    @staticmethod
    def liquidity_health(liquidity, fdv):
        ratio = liquidity / (fdv + 1e-6)
        return torch.clamp(ratio * 4.0, 0.0, 1.0)

    @staticmethod
    def buy_sell_imbalance(close, open_, high, low):
        range_hl = high - low + 1e-9
        body = close - open_
        strength = body / range_hl
        return torch.tanh(strength * 3.0)

    @staticmethod
    def fomo_acceleration(volume):
        vol_prev = torch.cat([volume[:, :1], volume[:, :-1]], dim=1)
        vol_chg = (volume - vol_prev) / (vol_prev + 1.0)
        vol_chg_prev = torch.cat([vol_chg[:, :1], vol_chg[:, :-1]], dim=1)
        acc = vol_chg - vol_chg_prev
        return torch.clamp(acc, -5.0, 5.0)

    @staticmethod
    def pump_deviation(close, window=20):
        pad = torch.zeros((close.shape[0], window-1), device=close.device)
        c_pad = torch.cat([pad, close], dim=1)
        ma = c_pad.unfold(1, window, 1).mean(dim=-1)
        dev = (close - ma) / (ma + 1e-9)
        return dev

    @staticmethod
    def volatility_clustering(close, window=10):
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        ret_sq = ret ** 2
        pad = torch.zeros((ret_sq.shape[0], window-1), device=close.device)
        ret_sq_pad = torch.cat([pad, ret_sq], dim=1)
        vol_ma = ret_sq_pad.unfold(1, window, 1).mean(dim=-1)
        return torch.sqrt(vol_ma + 1e-9)

    @staticmethod
    def momentum_reversal(close, window=5):
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        pad = torch.zeros((ret.shape[0], window-1), device=close.device)
        ret_pad = torch.cat([pad, ret], dim=1)
        mom = ret_pad.unfold(1, window, 1).sum(dim=-1)
        mom_prev = torch.roll(mom, 1, dims=1)
        reversal = (mom * mom_prev < 0).float()
        return reversal

    @staticmethod
    def relative_strength(close, window=14):
        ret = close - torch.roll(close, 1, dims=1)
        gains = torch.relu(ret)
        losses = torch.relu(-ret)
        pad = torch.zeros((gains.shape[0], window-1), device=close.device)
        gains_pad = torch.cat([pad, gains], dim=1)
        losses_pad = torch.cat([pad, losses], dim=1)
        avg_gain = gains_pad.unfold(1, window, 1).mean(dim=-1)
        avg_loss = losses_pad.unfold(1, window, 1).mean(dim=-1)
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return (rsi - 50) / 50

class FeatureEngineer:
    INPUT_DIM = 6
    @staticmethod
    def compute_features(raw_dict):
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        liq = raw_dict['liquidity']
        fdv = raw_dict['fdv']
        
        c_prev = torch.cat([c[:, :1], c[:, :-1]], dim=1)
        ret = torch.log(c / (c_prev + 1e-9))
        liq_score = MemeIndicators.liquidity_health(liq, fdv)
        pressure = MemeIndicators.buy_sell_imbalance(c, o, h, l)
        fomo = MemeIndicators.fomo_acceleration(v)
        dev = MemeIndicators.pump_deviation(c)
        log_vol = torch.log1p(v)
        
        def robust_norm(t, window=120):
            if t.dim() != 2 or t.shape[1] < 2:
                return torch.zeros_like(t)
            w = int(min(window, t.shape[1]))
            pad = t[:, :1].repeat(1, w - 1)
            t_pad = torch.cat([pad, t], dim=1)
            t_windows = t_pad.unfold(1, w, 1)
            median_val = torch.median(t_windows, dim=-1)[0]
            abs_diff = torch.abs(t_windows - median_val.unsqueeze(-1))
            mad_val = torch.median(abs_diff, dim=-1)[0] + 1e-6
            norm = (t - median_val) / mad_val
            norm = torch.clamp(norm, -5.0, 5.0)
            return torch.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)

        features = torch.stack([
            robust_norm(ret),
            liq_score,
            pressure,
            robust_norm(fomo),
            robust_norm(dev),
            robust_norm(log_vol)
        ], dim=1)
        return features

# ----------------------
# 5. 兼容性补丁 (Polyfills)
# ----------------------
if not hasattr(torch, 'roll'):
    def torch_roll(input, shifts, dims=None):
        if dims is None: dims = 0
        if isinstance(shifts, int): shifts = [shifts]
        if isinstance(dims, int): dims = [dims]
        result = input
        for s, d in zip(shifts, dims):
            if d < 0: d += input.dim()
            if s == 0: continue
            size = input.size(d)
            s = s % size
            if s == 0: continue
            head = input.narrow(d, size - s, s)
            tail = input.narrow(d, 0, size - s)
            result = torch.cat([head, tail], dim=d)
            input = result
        return result
    torch.roll = torch_roll

if not hasattr(torch, 'nanmedian'):
    def torch_nanmedian(input, dim=None, keepdim=False):
        if dim is None: input = input.flatten(); dim = 0
        filled = torch.where(torch.isnan(input), torch.zeros_like(input), input)
        return filled.median(dim=dim, keepdim=keepdim)
    torch.nanmedian = torch_nanmedian

if not hasattr(torch, 'nan_to_num'):
    def torch_nan_to_num(input, nan=0.0, posinf=None, neginf=None):
        posinf = nan if posinf is None else posinf
        neginf = nan if neginf is None else neginf
        x = torch.where(torch.isnan(input), torch.tensor(nan, device=input.device, dtype=input.dtype), input)
        x = torch.where(torch.isinf(x) & (x > 0), torch.tensor(posinf, device=input.device, dtype=input.dtype), x)
        x = torch.where(torch.isinf(x) & (x < 0), torch.tensor(neginf, device=input.device, dtype=input.dtype), x)
        return x
    torch.nan_to_num = torch_nan_to_num

if not hasattr(torch, 'argsort'):
    def torch_argsort(input, dim=-1, descending=False):
        _, indices = torch.sort(input, dim=dim, descending=descending)
        return indices
    torch.argsort = torch_argsort

# ----------------------
# 6. 策略逻辑 (StrategyMiner)
# ----------------------
class StrategyMiner:
    def __init__(self):
        self.best_formula_tokens = None
        self.token_mode = "postfix"
        self.formula_ready = False
        self.raw_formula_tokens = None

    def _is_valid_postfix(self, tokens):
        if not isinstance(tokens, (list, tuple)) or not tokens:
            return False
        stack_size = 0
        for t in tokens:
            try:
                t = int(t)
            except Exception:
                return False
            if t < len(_FEATURES):
                stack_size += 1
                continue
            arity = _OP_ARITY_MAP.get(t)
            if arity is None:
                return False
            arity = int(arity)
            if stack_size < arity:
                return False
            stack_size -= (arity - 1)
        return stack_size == 1

    def _is_valid_prefix(self, tokens):
        if not isinstance(tokens, (list, tuple)) or not tokens:
            return False
        stack_size = 0
        for t in reversed(tokens):
            try:
                t = int(t)
            except Exception:
                return False
            if t < len(_FEATURES):
                stack_size += 1
                continue
            arity = _OP_ARITY_MAP.get(t)
            if arity is None:
                return False
            arity = int(arity)
            if stack_size < arity:
                return False
            stack_size -= (arity - 1)
        return stack_size == 1

    def set_formula(self, tokens, mode="postfix"):
        self.token_mode = _normalize_token_mode(mode)
        try:
            toks = [int(t) for t in list(tokens)]
        except Exception:
            toks = [0]
        vocab_size = int(len(_FEATURES) + len(_OPS_CONFIG))
        toks = [t for t in toks if 0 <= int(t) < vocab_size]
        if not toks:
            toks = [0]
        self.raw_formula_tokens = list(toks)
        if self.token_mode == "prefix":
            if not self._is_valid_prefix(toks):
                toks = _repair_prefix(toks)
        else:
            if not self._is_valid_postfix(toks):
                toks = _repair_postfix(toks)
        self.best_formula_tokens = toks
        self.formula_ready = True
        expr = self.decode_expression(self.best_formula_tokens)
        log.info(f"单公式加载完成: {expr}")

    def ensure_formula(self):
        # 兼容旧逻辑，但在 set_formula 调用后通常不需要
        if not self.formula_ready:
            log.warn("公式未就绪")

    def decode_expression(self, tokens):
        mode = str(self.token_mode).lower()
        if mode == "prefix": return self.decode_prefix(tokens)
        return self.decode_postfix(tokens)

    def decode_prefix(self, tokens):
        if not tokens: return "N/A"
        stack = []
        symbol_map = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/"}
        for t in reversed(tokens):
            if t < len(_FEATURES):
                stack.append(_FEATURES[t])
                continue
            op_idx = t - len(_FEATURES)
            if op_idx < 0 or op_idx >= len(_OPS_CONFIG):
                stack.append(f"UNK_{t}")
                continue
            op_name, _, arity = _OPS_CONFIG[op_idx]
            if len(stack) < arity:
                return "INVALID"
            args = [stack.pop() for _ in range(arity)]
            if arity == 1:
                expr = f"(-{args[0]})" if op_name == "NEG" else f"{op_name}({args[0]})"
            elif arity == 2:
                expr = f"({args[0]} {symbol_map[op_name]} {args[1]})" if op_name in symbol_map else f"{op_name}({args[0]}, {args[1]})"
            else:
                expr = f"{op_name}({', '.join(args)})"
            stack.append(expr)
        return stack[-1] if len(stack) == 1 else "INVALID"

    def decode_postfix(self, tokens):
        if not tokens: return "N/A"
        stack = []
        symbol_map = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/"}
        for t in tokens:
            if t < len(_FEATURES):
                stack.append(_FEATURES[t])
                continue
            op_idx = t - len(_FEATURES)
            if op_idx < 0 or op_idx >= len(_OPS_CONFIG):
                stack.append(f"UNK_{t}")
                continue
            op_name, _, arity = _OPS_CONFIG[op_idx]
            if len(stack) < arity:
                return "INVALID"
            args = []
            for _ in range(arity): args.append(stack.pop())
            args.reverse()
            if arity == 1:
                expr = f"(-{args[0]})" if op_name == "NEG" else f"{op_name}({args[0]})"
            elif arity == 2:
                expr = f"({args[0]} {symbol_map[op_name]} {args[1]})" if op_name in symbol_map else f"{op_name}({args[0]}, {args[1]})"
            else:
                expr = f"{op_name}({', '.join(args)})"
            stack.append(expr)
        return stack[-1] if len(stack) == 1 else "INVALID"

    def execute_formula(self, formula_tokens, feat_tensor):
        if not formula_tokens:
            return None
        mode = str(self.token_mode).lower()
        default_val = feat_tensor[0]
        stack = []

        if mode == "prefix":
            seq = list(reversed(formula_tokens))
        else:
            seq = formula_tokens

        for token in seq:
            token = int(token)
            if token < len(_FEATURES):
                stack.append(feat_tensor[token])
                continue
            func = _OP_FUNC_MAP.get(token)
            if func is None:
                return None
            arity = int(_OP_ARITY_MAP.get(token, 0))
            if arity <= 0:
                return None
            if len(stack) < arity:
                missing = arity - len(stack)
                for _ in range(missing):
                    stack.append(default_val)
            args = []
            for _ in range(arity):
                args.append(stack.pop())
            if mode != "prefix":
                args.reverse()
            res = func(*args)
            if torch.isnan(res).any() or torch.isinf(res).any():
                res = torch.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            stack.append(res)

        if len(stack) == 1:
            return stack[0]
        return None

# ----------------------
# 7. 股票过滤与数据加载
# ----------------------
def filter_kcbj_stock(stock_list):
    return [s for s in stock_list if not (s.startswith('688') or s.startswith('8') or s.startswith('4'))]

def filter_st_stock(stock_list):
    current_data = get_current_data()
    return [s for s in stock_list if not current_data[s].is_st and 'ST' not in current_data[s].name and '*' not in current_data[s].name and '退' not in current_data[s].name]

def filter_paused_stock(stock_list):
    current_data = get_current_data()
    return [s for s in stock_list if not current_data[s].paused]

def _shift_trade_date_back(ref_date, n_days: int):
    if n_days <= 0:
        return ref_date
    try:
        days = get_trade_days(end_date=ref_date, count=n_days + 1)
        if days is None or len(days) == 0:
            return ref_date
        return days[0]
    except:
        return ref_date - timedelta(days=n_days)

def _get_lookback_start(end_date, lookback_trade_days: int):
    if lookback_trade_days <= 1:
        return end_date
    try:
        days = get_trade_days(end_date=end_date, count=lookback_trade_days)
        if days is None or len(days) == 0:
            return end_date
        return days[0]
    except:
        return end_date - timedelta(days=int(lookback_trade_days * 1.5))

def select_small_cap_stocks(context, ref_date=None):
    ref_date = context.previous_date if ref_date is None else ref_date
    try:
        initial = get_index_stocks(_BENCHMARK, ref_date)
    except:
        initial = []
    if not initial: return []
    initial = filter_kcbj_stock(initial)
    initial = filter_paused_stock(initial)
    return initial

class JQCrossSectionDataEngine:
    def __init__(self, stock_list, start_date, end_date):
        self.stocks = stock_list
        self.num_stocks = len(self.stocks)
        self.start_date = start_date
        self.end_date = end_date
        self.feat_data = None
        self.time_dim = None
        self.close_data = None

    def load(self):
        if not self.stocks: return self
        dfs = []
        for code in self.stocks:
            try:
                df = get_price(code, start_date=self.start_date, end_date=self.end_date, 
                             fields=['open', 'close', 'high', 'low', 'volume', 'money'], fq='pre')
                if df.empty: continue
                df['code'] = code
                df.index.name = 'date'
                dfs.append(df)
            except: continue
            
        if not dfs: return self
        df_all = pd.concat(dfs).sort_index()
        
        def get_tensor(col):
            try:
                p = df_all.reset_index().pivot_table(index='date', columns='code', values=col, aggfunc='last')
            except Exception:
                p = df_all.pivot(columns='code', values=col)
            for s in self.stocks:
                if s not in p.columns: p[s] = np.nan
            p = p[self.stocks].ffill().fillna(0)
            return torch.tensor(p.values.T, dtype=torch.float32, device=_DEVICE)
        
        try:
            raw_dict = {
                'open': get_tensor('open'),
                'high': get_tensor('high'),
                'low': get_tensor('low'),
                'close': get_tensor('close'),
                'volume': get_tensor('volume'),
                'liquidity': get_tensor('money'),
            }
            raw_dict['fdv'] = torch.zeros_like(raw_dict['close'])
            self.close_data = raw_dict['close']
            self.time_dim = raw_dict['close'].shape[1]
            if self.time_dim < 20: return self
            
            self.feat_data = FeatureEngineer.compute_features(raw_dict)
            if self.feat_data is not None and self.feat_data.dim() == 3:
                if self.feat_data.shape[0] == self.num_stocks and self.feat_data.shape[1] == len(_FEATURES):
                    self.feat_data = self.feat_data.permute(1, 0, 2).contiguous()
            if torch.isnan(self.feat_data).any():
                self.feat_data = torch.nan_to_num(self.feat_data, nan=0.0)
        except Exception as e:
            log.error(f"特征计算失败: {e}")
            
        return self

# ----------------------
# 8. 调仓主逻辑
# ----------------------
def rebalance(context):
    log.info(f"开始调仓: {context.current_dt}")
    
    # 1. 确定股票池
    end_date = _shift_trade_date_back(context.previous_date, 1)
    stocks = select_small_cap_stocks(context, ref_date=end_date)
    if not stocks:
        log.warn("股票池为空")
        return
        
    # 2. 准备数据
    start_date = _get_lookback_start(end_date, lookback_trade_days=120)
    
    engine = JQCrossSectionDataEngine(stocks, start_date, end_date).load()
    if engine.feat_data is None:
        log.warn("特征数据加载失败")
        return

    # 3. 计算因子
    context.miner.ensure_formula()
    if not context.miner.best_formula_tokens:
        log.warn("单公式为空，跳过调仓")
        return
        
    try:
        current_factor = context.miner.execute_formula(context.miner.best_formula_tokens, engine.feat_data)
    except Exception as e:
        log.error(f"公式执行异常: {e}")
        return
        
    if current_factor is None:
        log.warn("因子计算结果为空")
        return
        
    # 取最后一天的因子值
    current_factor = current_factor[:, -1]
    
    # 4. 排序选股
    scores = current_factor.detach().cpu().numpy().tolist()
    ranked = list(zip(scores, stocks))
    ranked.sort(
        key=lambda x: (
            0 if x[0] == x[0] else 1,
            -(x[0] if x[0] == x[0] else 0.0),
            str(x[1]),
        )
    )
    try:
        topk_cap = int(min(_TOP_K, len(stocks)))
        if topk_cap > 0:
            i = 0
            while i < len(ranked):
                v = ranked[i][0]
                j = i + 1
                while j < len(ranked) and ranked[j][0] == v:
                    j += 1
                seg_n = j - i
                if seg_n > 1 and i < topk_cap:
                    within_start = i + 1
                    within_end = min(j, topk_cap)
                    within_n = within_end - within_start + 1
                    if within_n > 1:
                        v_str = "NaN" if not (v == v) else (f"{float(v):.6g}" if isinstance(v, (int, float)) else str(v))
                        if j > topk_cap:
                            log.warn(
                                f"Top{topk_cap}并列分数影响入选: 排名{within_start}-{within_end}分数={v_str}；"
                                f"该分数全池并列{seg_n}名(排名{i + 1}-{j})，Top{topk_cap}落在并列组内，"
                                f"入选会受股票池顺序影响"
                            )
                        else:
                            log.warn(
                                f"Top{topk_cap}存在并列分数: 排名{within_start}-{within_end}分数={v_str}；"
                                f"该分数在Top{topk_cap}内并列{seg_n}名(排名{i + 1}-{j})，"
                                f"不影响是否入选，仅影响Top{topk_cap}内部顺序"
                            )
                i = j
    except Exception:
        pass
    target_stocks = [s for _, s in ranked[: min(_TOP_K, len(stocks))]]
    
    # 5. 执行交易
    current_data = get_current_data()
    total_value = float(context.portfolio.total_value)
    prices = {}
    for stock in target_stocks:
        try:
            px = float(current_data[stock].last_price)
        except:
            px = 0.0
        prices[stock] = px

    final_targets = []
    for k in range(len(target_stocks), 0, -1):
        budget_k = total_value / float(k)
        ok = True
        for s in target_stocks[:k]:
            px = prices.get(s, 0.0)
            if px <= 0.0 or budget_k / px < 100.0:
                ok = False
                break
        if ok:
            final_targets = target_stocks[:k]
            break

    if not final_targets:
        log.warn("资金不足以满足最小100股要求，跳过本次调仓")
        return

    current_holdings = list(context.portfolio.positions.keys())
    for stock in current_holdings:
        if stock not in final_targets:
            order_target(stock, 0)

    value_per_stock = float(context.portfolio.total_value) / float(len(final_targets))
    for s in final_targets:
        try:
            price = float(current_data[s].last_price)
        except:
            price = 0.0
        if price <= 0:
            continue
        target_shares = int(value_per_stock / price / 100.0) * 100
        if target_shares < 100:
            target_shares = 0

        current_position = context.portfolio.positions[s] if s in context.portfolio.positions else None
        current_shares = int(current_position.total_amount) if current_position is not None else 0
        delta = target_shares - current_shares
        if abs(delta) < 100 and target_shares != 0:
            continue
        if target_shares != current_shares:
            order_target(s, target_shares)
    
    log.info(f"调仓完成，持有: {final_targets}")
