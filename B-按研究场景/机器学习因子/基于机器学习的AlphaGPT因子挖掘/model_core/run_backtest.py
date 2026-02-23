import warnings
import os
import pandas as pd
import numpy as np
import logging
import contextlib
os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
from .qlib_loader import _filter_kcbj_instruments

_DEVNULL = None


def _devnull_stream():
    dn = globals().get("_DEVNULL")
    if dn is None or getattr(dn, "closed", False):
        dn = open(os.devnull, "w")
        globals()["_DEVNULL"] = dn
    return dn


def _want_factor(mode):
    return str(mode).lower() in {"pre_adjusted", "pre", "adjusted"}

def _want_raw(mode):
    return str(mode).lower() in {"raw", "real", "unadjusted"}


def _coerce_signal_df(signal_df=None, pred_df=None):
    if signal_df is not None:
        out = signal_df.copy()
        out.index = pd.to_datetime(out.index)
        return out

    if pred_df is None:
        raise ValueError("signal_df or pred_df is required")

    if isinstance(pred_df, pd.DataFrame):
        if "score" in pred_df.columns:
            s = pred_df["score"]
        elif pred_df.shape[1] == 1:
            s = pred_df.iloc[:, 0]
        else:
            raise ValueError("pred_df must contain a 'score' column or be single-column")
    elif isinstance(pred_df, pd.Series):
        s = pred_df
    else:
        raise ValueError("pred_df must be a pandas DataFrame or Series")

    if not isinstance(s.index, pd.MultiIndex) or s.index.nlevels < 2:
        raise ValueError("pred_df must be indexed by (datetime, instrument)")

    if s.index.names and "datetime" in s.index.names:
        out = s.unstack("instrument")
    else:
        out = s.unstack()
    out.index = pd.to_datetime(out.index)
    return out


def _align_signal_df(signal_df, dates, asset_list):
    out = signal_df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.reindex(index=pd.to_datetime(dates)).ffill()
    out = out.reindex(columns=list(asset_list))
    return out


def _rank_targets(scores: pd.Series, topk: int, pool_order=None):
    if scores is None:
        return []
    try:
        s = scores.dropna()
    except Exception:
        s = scores
    if s is None or len(s) == 0:
        return []
    df = s.to_frame("score")
    if pool_order is not None:
        try:
            pos = {str(sym): i for i, sym in enumerate(list(pool_order))}
            df["pool_pos"] = df.index.astype(str).map(lambda x: pos.get(str(x), 10**18))
            df = df.sort_values(["score", "pool_pos"], ascending=[False, True], kind="mergesort")
        except Exception:
            df["instrument"] = df.index.astype(str)
            df = df.sort_values(["score", "instrument"], ascending=[False, True], kind="mergesort")
    else:
        df["instrument"] = df.index.astype(str)
        df = df.sort_values(["score", "instrument"], ascending=[False, True], kind="mergesort")
    return df.head(int(topk)).index.tolist()


def run_qlib_backtest(
    signal_df=None,
    pred_df=None,
    start_time='2022-01-01',
    end_time='2023-12-31',
    instruments='csi300',
    topk=10,
    benchmark='SH000300',
    account=1000000,
    provider_uri='/Users/shuyan/.qlib/qlib_data/cn_data/qlib_bin',
    signal_lag=0,
    skip_months=(),
    commission_rate=0.0003,
    stamp_tax_rate=0.001,
    min_commission=5.0,
    lot_size=100,
    slippage_rate=0.003,
    target_value_per_stock=100000.0,
    buy_limit_multiplier=1.02,
    sell_limit_multiplier=0.98,
    strict_data_coverage=False,
    lookback_days=730,
    price_mode="pre_adjusted",
    trade_price_mode=None,
    ref_price_mode=None,
    benchmark_price_mode="raw",
    return_trades=False,
    return_rebalances=False,
):
    warnings.filterwarnings("ignore", category=UserWarning, module="gym")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="qlib.utils.index_data")
    logging.getLogger("qlib").setLevel(logging.ERROR)
    logging.getLogger("qlib").propagate = False
    logging.getLogger("qlib.backtest").setLevel(logging.ERROR)
    logging.getLogger("qlib.backtest").propagate = False
    logging.getLogger("qlib.backtest.exchange").setLevel(logging.ERROR)
    logging.getLogger("qlib.backtest.exchange").propagate = False
    logging.getLogger("gym").setLevel(logging.ERROR)
    dn = _devnull_stream()
    with contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        import qlib
        from qlib.data import D as QlibData
        from qlib.constant import REG_CN as QlibRegion
        from qlib.config import C
        from qlib.contrib.evaluate import risk_analysis
        qlib.init(provider_uri=provider_uri, region=QlibRegion)
        C.kernels = 1
        C.joblib_backend = "threading"

    trade_start = pd.to_datetime(start_time)
    trade_end = pd.to_datetime(end_time)
    load_start = (trade_start - pd.Timedelta(days=int(lookback_days))).strftime("%Y-%m-%d") if lookback_days else start_time

    if trade_price_mode is None:
        trade_price_mode = price_mode
    if ref_price_mode is None:
        ref_price_mode = trade_price_mode

    instruments_provider = QlibData.instruments(market=instruments)
    pool_cache = {}

    def _pool_on(dt):
        if not callable(instruments_provider):
            return None
        try:
            key = pd.to_datetime(dt).normalize()
        except Exception:
            key = dt
        if key in pool_cache:
            return pool_cache[key]
        try:
            pool = instruments_provider(key)
            if pool is None:
                pool_cache[key] = None
            else:
                pool_cache[key] = list(map(str, list(pool)))
        except Exception:
            pool_cache[key] = None
        return pool_cache[key]

    want_factor = _want_factor(price_mode) or _want_factor(trade_price_mode) or _want_factor(ref_price_mode) or _want_factor(benchmark_price_mode)
    px_fields = ['$open', '$high', '$low', '$close', '$volume']
    if want_factor:
        px_fields.append('$factor')
    px_fields.extend(['$high_limit', '$low_limit'])
    raw_df = QlibData.features(instruments_provider, px_fields, start_time=load_start, end_time=end_time)

    if raw_df.index.names == ['instrument', 'datetime']:
        open_px = raw_df['$open'].unstack(level='datetime')
        high_px = raw_df['$high'].unstack(level='datetime')
        low_px = raw_df['$low'].unstack(level='datetime')
        close_px = raw_df['$close'].unstack(level='datetime')
        volume_px = raw_df['$volume'].unstack(level='datetime') if '$volume' in raw_df.columns else None
        factor_px = raw_df['$factor'].unstack(level='datetime') if '$factor' in raw_df.columns else None
        high_limit_px = raw_df['$high_limit'].unstack(level='datetime') if '$high_limit' in raw_df.columns else None
        low_limit_px = raw_df['$low_limit'].unstack(level='datetime') if '$low_limit' in raw_df.columns else None
    else:
        open_px = raw_df['$open'].unstack(level='instrument').T
        high_px = raw_df['$high'].unstack(level='instrument').T
        low_px = raw_df['$low'].unstack(level='instrument').T
        close_px = raw_df['$close'].unstack(level='instrument').T
        volume_px = raw_df['$volume'].unstack(level='instrument').T if '$volume' in raw_df.columns else None
        factor_px = raw_df['$factor'].unstack(level='instrument').T if '$factor' in raw_df.columns else None
        high_limit_px = raw_df['$high_limit'].unstack(level='instrument').T if '$high_limit' in raw_df.columns else None
        low_limit_px = raw_df['$low_limit'].unstack(level='instrument').T if '$low_limit' in raw_df.columns else None

    open_px.columns = pd.to_datetime(open_px.columns)
    high_px.columns = pd.to_datetime(high_px.columns)
    low_px.columns = pd.to_datetime(low_px.columns)
    close_px.columns = pd.to_datetime(close_px.columns)
    open_px = open_px.sort_index(axis=1)
    high_px = high_px.reindex(columns=open_px.columns)
    low_px = low_px.reindex(columns=open_px.columns)
    close_px = close_px.reindex(columns=open_px.columns)

    if bool(strict_data_coverage):
        try:
            valid_close = close_px.where(close_px > 0).notna()
            valid_arr = valid_close.to_numpy(dtype=bool, copy=False)
            rev = valid_arr[:, ::-1]
            has = rev.any(axis=1)
            last_from_end = rev.argmax(axis=1)
            last_pos = np.where(has, valid_arr.shape[1] - 1 - last_from_end, -1)
            cols = close_px.columns.to_numpy()
            last_dates = pd.to_datetime(np.where(last_pos >= 0, cols[last_pos], np.datetime64("NaT")))
            cutoff = trade_end - pd.Timedelta(days=5)
            keep_mask = last_dates >= cutoff
            if bool(keep_mask.any()) and int(keep_mask.sum()) < int(len(keep_mask)):
                keep_idx = close_px.index[keep_mask]
                open_px = open_px.loc[keep_idx]
                high_px = high_px.loc[keep_idx]
                low_px = low_px.loc[keep_idx]
                close_px = close_px.loc[keep_idx]
                if volume_px is not None:
                    volume_px = volume_px.loc[keep_idx]
                if factor_px is not None:
                    factor_px = factor_px.loc[keep_idx]
                if high_limit_px is not None:
                    high_limit_px = high_limit_px.loc[keep_idx]
                if low_limit_px is not None:
                    low_limit_px = low_limit_px.loc[keep_idx]
        except Exception:
            pass

    open_px = open_px.ffill(axis=1)
    high_px = high_px.ffill(axis=1)
    low_px = low_px.ffill(axis=1)
    close_px = close_px.ffill(axis=1)
    if volume_px is not None:
        volume_px.columns = pd.to_datetime(volume_px.columns)
        volume_px = volume_px.reindex(columns=open_px.columns).fillna(0.0)
    if factor_px is not None:
        factor_px.columns = pd.to_datetime(factor_px.columns)
        factor_px = factor_px.reindex(columns=open_px.columns).ffill(axis=1).fillna(1.0)
        open_px_trade = open_px * factor_px if _want_factor(trade_price_mode) else open_px
        high_px_trade = high_px * factor_px if _want_factor(trade_price_mode) else high_px
        low_px_trade = low_px * factor_px if _want_factor(trade_price_mode) else low_px
        close_px_trade = close_px * factor_px if _want_factor(trade_price_mode) else close_px
        close_px_ref = close_px * factor_px if _want_factor(ref_price_mode) else close_px
    else:
        open_px_trade = open_px
        high_px_trade = high_px
        low_px_trade = low_px
        close_px_trade = close_px
        close_px_ref = close_px
    if high_limit_px is not None:
        high_limit_px.columns = pd.to_datetime(high_limit_px.columns)
        high_limit_px = high_limit_px.reindex(columns=open_px.columns)
        if factor_px is not None and _want_factor(trade_price_mode):
            high_limit_px = high_limit_px * factor_px
        if not bool(high_limit_px.notna().any().any()):
            high_limit_px = None
    if low_limit_px is not None:
        low_limit_px.columns = pd.to_datetime(low_limit_px.columns)
        low_limit_px = low_limit_px.reindex(columns=open_px.columns)
        if factor_px is not None and _want_factor(trade_price_mode):
            low_limit_px = low_limit_px * factor_px
        if not bool(low_limit_px.notna().any().any()):
            low_limit_px = None

    asset_list = open_px_trade.index
    dates = open_px_trade.columns
    trade_mask = (dates >= trade_start) & (dates <= trade_end)
    trade_dates = dates[trade_mask]

    limit_pct = pd.Series(0.10, index=asset_list, dtype="float64")
    try:
        idx = limit_pct.index.astype(str)
        is_20 = idx.str.startswith("SH688") | idx.str.startswith("SZ300") | idx.str.startswith("SZ301")
        limit_pct.loc[is_20] = 0.20
    except Exception:
        limit_pct = limit_pct

    month_key = trade_dates.to_period("M")
    rebalance_dates = pd.Series(trade_dates, index=trade_dates).groupby(month_key).min().values
    rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    signal_df = _coerce_signal_df(signal_df=signal_df, pred_df=pred_df)
    signal_df = _align_signal_df(signal_df, dates=dates, asset_list=asset_list)
    pred_df = signal_df.stack().to_frame("score")
    pred_df.index.names = ["datetime", "instrument"]

    shares = pd.Series(0.0, index=asset_list, dtype="float64")
    cash = float(account)
    prev_close_value = float(account)
    report_rows = []
    positions_rows = []
    trade_rows = []
    rebalance_rows = []
    rebalance_set = set(rebalance_dates)
    skip_set = set(skip_months)
    date_pos = {d: i for i, d in enumerate(dates)}

    def _round_px(x):
        try:
            return float(round(float(x) + 1e-12, 2))
        except Exception:
            return float(x) if x is not None else 0.0

    for d in trade_dates:
        i = date_pos.get(d)
        open_prices = open_px_trade[d]
        high_prices = high_px_trade[d] if high_px_trade is not None else None
        low_prices = low_px_trade[d] if low_px_trade is not None else None
        close_prices = close_px_trade[d]
        open_prices = open_prices.where(open_prices > 0)
        if high_prices is not None:
            high_prices = high_prices.where(high_prices > 0)
        if low_prices is not None:
            low_prices = low_prices.where(low_prices > 0)
        close_prices = close_prices.where(close_prices > 0)
        volumes = volume_px[d] if volume_px is not None else None
        if volumes is not None:
            volumes = volumes.where(volumes > 0)
        high_limits = high_limit_px[d] if high_limit_px is not None else None
        low_limits = low_limit_px[d] if low_limit_px is not None else None
        slip = (float(slippage_rate) / 2.0) if slippage_rate else 0.0

        prev_close_for_limits = None
        if i is not None and (i - 1) >= 0:
            try:
                prev_close_for_limits = close_px_trade[dates[i - 1]]
            except Exception:
                prev_close_for_limits = None
        if prev_close_for_limits is not None:
            computed_hl = (prev_close_for_limits * (1.0 + limit_pct)).round(2)
            computed_ll = (prev_close_for_limits * (1.0 - limit_pct)).round(2)
            if high_limits is None:
                high_limits = computed_hl
            else:
                try:
                    high_limits = high_limits.fillna(computed_hl)
                except Exception:
                    high_limits = high_limits
            if low_limits is None:
                low_limits = computed_ll
            else:
                try:
                    low_limits = low_limits.fillna(computed_ll)
                except Exception:
                    low_limits = low_limits

        buy_block = None
        try:
            if high_limits is not None:
                buy_block = open_prices.round(2) >= high_limits.round(2)
            elif high_prices is not None:
                buy_block = (open_prices.round(2) == high_prices.round(2)) & (close_prices.round(2) == high_prices.round(2))
        except Exception:
            buy_block = None
        sell_block = None
        try:
            if low_limits is not None:
                sell_block = open_prices.round(2) <= low_limits.round(2)
            elif low_prices is not None:
                sell_block = (open_prices.round(2) == low_prices.round(2)) & (close_prices.round(2) == low_prices.round(2))
        except Exception:
            sell_block = None

        turnover = 0.0
        total_cost = 0.0

        if d in rebalance_set:
            if i is None:
                continue
            prev_i = i - 1 - int(signal_lag)
            targets = []
            if prev_i >= 0:
                prev_d = dates[prev_i]
                scores = signal_df.loc[prev_d]
                pool = _pool_on(prev_d)
                if pool is not None:
                    try:
                        scores = scores.reindex(pool).fillna(0.0)
                    except Exception:
                        scores = scores
                targets = _rank_targets(scores, topk=topk, pool_order=pool)

            if d.month in skip_set:
                targets = []

            ref_prices = None
            prev_trade_i = i - 1
            if prev_trade_i >= 0:
                ref_prices = close_px_ref[dates[prev_trade_i]]

            current_holdings = shares[shares > 0].index.tolist()
            sell_list = [s for s in current_holdings if s not in targets]
            min_trade_shares = float(lot_size) if lot_size else 0.0

            for s in sell_list:
                px_open = float(open_prices.get(s, 0.0))
                if not (px_open > 0):
                    continue
                if volumes is not None and not (float(volumes.get(s, 0.0)) > 0):
                    continue
                if sell_block is not None and bool(sell_block.get(s, False)):
                    continue
                if low_limits is not None and not (px_open > float(low_limits.get(s, 0.0)) + 1e-9):
                    continue
                cur = float(shares.get(s, 0.0))
                if not (cur > 0):
                    continue
                sell_shares = cur
                if lot_size and lot_size > 1:
                    sell_shares = (sell_shares // float(lot_size)) * float(lot_size)
                if not (sell_shares > 0):
                    continue
                trade_px = _round_px(px_open * (1.0 - slip)) if slip > 0 else float(px_open)
                if not (trade_px > 0):
                    continue
                if low_limits is not None and not (trade_px > float(low_limits.get(s, 0.0)) + 1e-9):
                    continue
                ref_px = float(ref_prices.get(s, 0.0)) if ref_prices is not None else 0.0
                if not (ref_px > 0):
                    ref_px = float(px_open)
                sell_limit_px = _round_px(ref_px * float(sell_limit_multiplier))
                if trade_px < sell_limit_px - 1e-9:
                    continue

                trade_value = sell_shares * trade_px
                comm = max(float(min_commission), trade_value * float(commission_rate)) if commission_rate and trade_value > 0 else 0.0
                tax = trade_value * float(stamp_tax_rate) if stamp_tax_rate else 0.0
                cost = comm + tax
                cash += trade_value - cost
                turnover += trade_value
                total_cost += cost
                shares.loc[s] = max(0.0, cur - sell_shares)
                trade_rows.append(
                    {
                        "datetime": d,
                        "action": "SELL",
                        "instrument": s,
                        "shares": float(sell_shares),
                        "price": float(trade_px),
                        "trade_value": float(trade_value),
                        "commission": float(comm),
                        "tax": float(tax),
                        "slippage_rate": float(slip),
                        "cash_after": float(cash),
                    }
                )

            if targets:
                target_shares_map = {}
                for s in targets:
                    px_open = float(open_prices.get(s, 0.0))
                    if not (px_open > 0):
                        target_shares_map[str(s)] = 0.0
                        continue
                    ref_px = float(ref_prices.get(s, 0.0)) if ref_prices is not None else 0.0
                    if not (ref_px > 0):
                        ref_px = float(px_open)
                    if target_value_per_stock is None:
                        try:
                            valid_px = open_prices.notna() & (open_prices > 0)
                            port_value = float(cash) + float((shares[valid_px] * open_prices[valid_px]).sum())
                        except Exception:
                            port_value = float(cash)
                        per_value = float(port_value) / float(len(targets)) if targets else 0.0
                    else:
                        per_value = float(target_value_per_stock)
                    desired = float(per_value) / float(ref_px) if float(ref_px) > 0 else 0.0
                    if lot_size and lot_size > 1:
                        desired = (desired // float(lot_size)) * float(lot_size)
                    desired = float(max(desired, 0.0))
                    if min_trade_shares > 0 and desired < min_trade_shares:
                        desired = 0.0
                    target_shares_map[str(s)] = desired

                for s in targets:
                    px_open = float(open_prices.get(s, 0.0))
                    if not (px_open > 0):
                        continue
                    if volumes is not None and not (float(volumes.get(s, 0.0)) > 0):
                        continue
                    if buy_block is not None and bool(buy_block.get(s, False)):
                        continue
                    if sell_block is not None and bool(sell_block.get(s, False)):
                        continue
                    cur = float(shares.get(s, 0.0))
                    tgt = float(target_shares_map.get(str(s), 0.0))
                    delta = tgt - cur
                    if min_trade_shares > 0 and abs(delta) < min_trade_shares and tgt != 0:
                        continue
                    if delta >= 0:
                        continue
                    if low_limits is not None and not (px_open > float(low_limits.get(s, 0.0)) + 1e-9):
                        continue
                    sell_shares = -delta
                    if lot_size and lot_size > 1:
                        sell_shares = (sell_shares // float(lot_size)) * float(lot_size)
                    sell_shares = min(sell_shares, cur)
                    if not (sell_shares > 0):
                        continue
                    trade_px = _round_px(px_open * (1.0 - slip)) if slip > 0 else float(px_open)
                    if low_limits is not None and not (trade_px > float(low_limits.get(s, 0.0)) + 1e-9):
                        continue
                    ref_px = float(ref_prices.get(s, 0.0)) if ref_prices is not None else 0.0
                    if not (ref_px > 0):
                        ref_px = float(px_open)
                    sell_limit_px = _round_px(ref_px * float(sell_limit_multiplier))
                    if trade_px < sell_limit_px - 1e-9:
                        continue
                    trade_value = sell_shares * trade_px
                    comm = max(float(min_commission), trade_value * float(commission_rate)) if commission_rate and trade_value > 0 else 0.0
                    tax = trade_value * float(stamp_tax_rate) if stamp_tax_rate else 0.0
                    cost = comm + tax
                    cash += trade_value - cost
                    turnover += trade_value
                    total_cost += cost
                    shares.loc[s] = max(0.0, cur - sell_shares)
                    trade_rows.append(
                        {
                            "datetime": d,
                            "action": "SELL",
                            "instrument": s,
                            "shares": float(sell_shares),
                            "price": float(trade_px),
                            "trade_value": float(trade_value),
                            "commission": float(comm),
                            "tax": float(tax),
                            "slippage_rate": float(slip),
                            "cash_after": float(cash),
                        }
                    )

                for s in targets:
                    px_open = float(open_prices.get(s, 0.0))
                    if not (px_open > 0):
                        continue
                    if volumes is not None and not (float(volumes.get(s, 0.0)) > 0):
                        continue
                    if buy_block is not None and bool(buy_block.get(s, False)):
                        continue
                    if sell_block is not None and bool(sell_block.get(s, False)):
                        continue
                    if high_limits is not None and not (px_open < float(high_limits.get(s, 0.0)) - 1e-9):
                        continue
                    cur = float(shares.get(s, 0.0))
                    tgt = float(target_shares_map.get(str(s), 0.0))
                    delta = tgt - cur
                    if min_trade_shares > 0 and abs(delta) < min_trade_shares and tgt != 0:
                        continue
                    if delta <= 0:
                        continue
                    buy_shares = delta
                    if lot_size and lot_size > 1:
                        buy_shares = (buy_shares // float(lot_size)) * float(lot_size)
                    if not (buy_shares > 0):
                        continue
                    trade_px = _round_px(px_open * (1.0 + slip)) if slip > 0 else float(px_open)
                    if high_limits is not None and not (trade_px < float(high_limits.get(s, 0.0)) - 1e-9):
                        continue
                    ref_px = float(ref_prices.get(s, 0.0)) if ref_prices is not None else 0.0
                    if not (ref_px > 0):
                        ref_px = float(px_open)
                    buy_limit_px = _round_px(ref_px * float(buy_limit_multiplier))
                    if trade_px > buy_limit_px + 1e-9:
                        continue

                    trade_value = buy_shares * trade_px
                    comm = max(float(min_commission), trade_value * float(commission_rate)) if commission_rate and trade_value > 0 else 0.0
                    spend = trade_value + comm
                    if spend > cash:
                        continue
                    cash -= spend
                    turnover += trade_value
                    total_cost += comm
                    shares.loc[s] = cur + buy_shares
                    trade_rows.append(
                        {
                            "datetime": d,
                            "action": "BUY",
                            "instrument": s,
                            "shares": float(buy_shares),
                            "price": float(trade_px),
                            "trade_value": float(trade_value),
                            "commission": float(comm),
                            "tax": 0.0,
                            "slippage_rate": float(slip),
                            "cash_after": float(cash),
                        }
                    )

                cash = float(max(cash, 0.0))
                holdings_after = shares[shares > 0].index.tolist()
                rebalance_rows.append(
                    {
                        "datetime": d,
                        "targets": ",".join(list(map(str, targets))) if targets else "",
                        "holdings_after": ",".join(list(map(str, holdings_after))) if holdings_after else "",
                        "cash_after": float(cash),
                    }
                )

        mark_close = close_prices
        try:
            mark_close = mark_close.where(mark_close > 0)
        except Exception:
            mark_close = mark_close
        valid_close = mark_close.notna()
        close_value = cash + float((shares[valid_close] * mark_close[valid_close]).sum())
        daily_ret = (close_value / prev_close_value - 1.0) if prev_close_value > 0 else 0.0
        prev_close_value = close_value

        report_rows.append(
            {
                "datetime": d,
                "account": close_value,
                "return": daily_ret,
                "total_turnover": turnover,
                "total_cost": total_cost,
                "cash": cash,
            }
        )
        positions_rows.append(pd.Series(shares.values, index=asset_list, name=d))

    report_df = pd.DataFrame(report_rows).set_index("datetime")
    positions_df = pd.DataFrame(positions_rows)
    positions_df.index = pd.to_datetime(positions_df.index)

    bench_ret = pd.Series(0.0, index=report_df.index, dtype="float64")
    try:
        bench_fields = ['$close']
        if _want_factor(benchmark_price_mode):
            bench_fields.append('$factor')
        first_day = pd.to_datetime(report_df.index.min())
        load_start = (first_day - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        bench_df = QlibData.features([benchmark], bench_fields, start_time=load_start, end_time=end_time)
        if bench_df.index.names == ['instrument', 'datetime']:
            bench_close = bench_df['$close'].unstack(level='datetime').iloc[0]
            bench_factor = bench_df['$factor'].unstack(level='datetime').iloc[0] if '$factor' in bench_df.columns else None
        else:
            bench_close = bench_df['$close'].unstack(level='instrument').iloc[:, 0]
            bench_factor = bench_df['$factor'].unstack(level='instrument').iloc[:, 0] if '$factor' in bench_df.columns else None
        bench_close.index = pd.to_datetime(bench_close.index)
        if bench_factor is not None and _want_factor(benchmark_price_mode):
            bench_factor.index = pd.to_datetime(bench_factor.index)
            bench_close = bench_close * bench_factor
        prev_days = bench_close.index[bench_close.index < first_day]
        if len(prev_days) > 0:
            prev_day = prev_days.max()
        else:
            prev_day = first_day
        full_index = pd.DatetimeIndex([prev_day]).append(pd.to_datetime(report_df.index))
        bench_close = bench_close.reindex(full_index).ffill()
        bench_ret = bench_close.pct_change().reindex(pd.to_datetime(report_df.index)).fillna(0.0)
    except Exception:
        bench_ret = bench_ret

    report_df["bench"] = bench_ret
    analysis = risk_analysis(report_df["return"] - report_df["bench"])
    abs_analysis = risk_analysis(report_df["return"])
    trades_df = pd.DataFrame(trade_rows)
    rebalances_df = pd.DataFrame(rebalance_rows).set_index("datetime") if rebalance_rows else pd.DataFrame(columns=["targets", "holdings_after", "cash_after"]).set_index(pd.DatetimeIndex([], name="datetime"))
    if bool(return_trades) and bool(return_rebalances):
        return report_df, positions_df, pred_df, analysis, abs_analysis, trades_df, rebalances_df
    if bool(return_trades):
        return report_df, positions_df, pred_df, analysis, abs_analysis, trades_df
    if bool(return_rebalances):
        return report_df, positions_df, pred_df, analysis, abs_analysis, rebalances_df
    return report_df, positions_df, pred_df, analysis, abs_analysis


def run_joinquant_backtest(
    signal_df=None,
    pred_df=None,
    start_time='2022-01-01',
    end_time='2023-12-31',
    instruments='csi300',
    topk=10,
    benchmark='SH000300',
    account=1000000,
    provider_uri='/Users/shuyan/.qlib/qlib_data/cn_data/qlib_bin',
    signal_lag=0,
    rebalance_n_days=None,
    skip_months=(),
    commission_rate=0.0003,
    stamp_tax_rate=0.001,
    min_commission=5.0,
    lot_size=100,
    slippage_rate=0.003,
    target_value_per_stock=100000.0,
    buy_limit_multiplier=None,
    sell_limit_multiplier=None,
    strict_data_coverage=False,
    lookback_days=730,
    backfill_untradable=False,
    trade_price_mode="raw",
    ref_price_mode="raw",
    benchmark_price_mode="raw",
    return_trades=False,
    return_rebalances=False,
    targets_override=None,
    target_shares_override=None,
    apply_dividend_cash=True,
    corporate_action_mode="cash",
):
    warnings.filterwarnings("ignore", category=UserWarning, module="gym")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="qlib.utils.index_data")
    logging.getLogger("qlib").setLevel(logging.ERROR)
    logging.getLogger("qlib").propagate = False
    logging.getLogger("qlib.backtest").setLevel(logging.ERROR)
    logging.getLogger("qlib.backtest").propagate = False
    logging.getLogger("qlib.backtest.exchange").setLevel(logging.ERROR)
    logging.getLogger("qlib.backtest.exchange").propagate = False
    logging.getLogger("gym").setLevel(logging.ERROR)
    dn = _devnull_stream()
    with contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        import qlib
        from qlib.data import D as QlibData
        from qlib.constant import REG_CN as QlibRegion
        from qlib.config import C
        from qlib.contrib.evaluate import risk_analysis
        qlib.init(provider_uri=provider_uri, region=QlibRegion)
        C.kernels = 1
        C.joblib_backend = "threading"

    trade_start = pd.to_datetime(start_time)
    trade_end = pd.to_datetime(end_time)
    load_start = (trade_start - pd.Timedelta(days=int(lookback_days))).strftime("%Y-%m-%d") if lookback_days else start_time

    instruments_provider = QlibData.instruments(market=instruments)
    pool_cache = {}

    def _pool_on(dt):
        try:
            key = pd.to_datetime(dt).normalize()
        except Exception:
            key = dt
        if key in pool_cache:
            return pool_cache.get(key)
        pool = None
        if callable(instruments_provider):
            try:
                pool = instruments_provider(key)
            except Exception:
                pool = None
        if pool is None:
            try:
                dt_str = pd.to_datetime(key).strftime("%Y-%m-%d")
                pool = QlibData.list_instruments(instruments=instruments_provider, start_time=dt_str, end_time=dt_str, as_list=True)
            except Exception:
                pool = None
        if pool is not None:
            try:
                pool = [str(x) for x in list(pool) if str(x).strip()]
                pool = _filter_kcbj_instruments(pool)
                pool = sorted(pool)
            except Exception:
                pool = pool
        pool_cache[key] = pool
        return pool
    px_fields = ['$open', '$high', '$low', '$close', '$volume', '$factor']
    px_fields.extend(['$high_limit', '$low_limit'])
    raw_df = QlibData.features(instruments_provider, px_fields, start_time=load_start, end_time=end_time)

    if raw_df.index.names == ['instrument', 'datetime']:
        open_px = raw_df['$open'].unstack(level='datetime')
        high_px = raw_df['$high'].unstack(level='datetime')
        low_px = raw_df['$low'].unstack(level='datetime')
        close_px = raw_df['$close'].unstack(level='datetime')
        volume_px = raw_df['$volume'].unstack(level='datetime') if '$volume' in raw_df.columns else None
        factor_px = raw_df['$factor'].unstack(level='datetime') if '$factor' in raw_df.columns else None
        high_limit_px = raw_df['$high_limit'].unstack(level='datetime') if '$high_limit' in raw_df.columns else None
        low_limit_px = raw_df['$low_limit'].unstack(level='datetime') if '$low_limit' in raw_df.columns else None
    else:
        open_px = raw_df['$open'].unstack(level='instrument').T
        high_px = raw_df['$high'].unstack(level='instrument').T
        low_px = raw_df['$low'].unstack(level='instrument').T
        close_px = raw_df['$close'].unstack(level='instrument').T
        volume_px = raw_df['$volume'].unstack(level='instrument').T if '$volume' in raw_df.columns else None
        factor_px = raw_df['$factor'].unstack(level='instrument').T if '$factor' in raw_df.columns else None
        high_limit_px = raw_df['$high_limit'].unstack(level='instrument').T if '$high_limit' in raw_df.columns else None
        low_limit_px = raw_df['$low_limit'].unstack(level='instrument').T if '$low_limit' in raw_df.columns else None

    open_px.columns = pd.to_datetime(open_px.columns)
    high_px.columns = pd.to_datetime(high_px.columns)
    low_px.columns = pd.to_datetime(low_px.columns)
    close_px.columns = pd.to_datetime(close_px.columns)
    open_px = open_px.sort_index(axis=1)
    high_px = high_px.reindex(columns=open_px.columns)
    low_px = low_px.reindex(columns=open_px.columns)
    close_px = close_px.reindex(columns=open_px.columns)
    if bool(strict_data_coverage):
        try:
            valid_close = close_px.where(close_px > 0).notna()
            valid_arr = valid_close.to_numpy(dtype=bool, copy=False)
            rev = valid_arr[:, ::-1]
            has = rev.any(axis=1)
            last_from_end = rev.argmax(axis=1)
            last_pos = np.where(has, valid_arr.shape[1] - 1 - last_from_end, -1)
            cols = close_px.columns.to_numpy()
            last_dates = pd.to_datetime(np.where(last_pos >= 0, cols[last_pos], np.datetime64("NaT")))
            cutoff = trade_end - pd.Timedelta(days=5)
            keep_mask = last_dates >= cutoff
            if bool(keep_mask.any()) and int(keep_mask.sum()) < int(len(keep_mask)):
                keep_idx = close_px.index[keep_mask]
                open_px = open_px.loc[keep_idx]
                high_px = high_px.loc[keep_idx]
                low_px = low_px.loc[keep_idx]
                close_px = close_px.loc[keep_idx]
                if volume_px is not None:
                    volume_px = volume_px.loc[keep_idx]
                if factor_px is not None:
                    factor_px = factor_px.loc[keep_idx]
                if high_limit_px is not None:
                    high_limit_px = high_limit_px.loc[keep_idx]
                if low_limit_px is not None:
                    low_limit_px = low_limit_px.loc[keep_idx]
        except Exception:
            pass
    if volume_px is not None:
        volume_px.columns = pd.to_datetime(volume_px.columns)
        volume_px = volume_px.reindex(columns=open_px.columns)
    if factor_px is not None:
        factor_px.columns = pd.to_datetime(factor_px.columns)
        factor_px = factor_px.reindex(columns=open_px.columns)
    if high_limit_px is not None:
        high_limit_px.columns = pd.to_datetime(high_limit_px.columns)
        high_limit_px = high_limit_px.reindex(columns=open_px.columns)
        if not bool(high_limit_px.notna().any().any()):
            high_limit_px = None
    if low_limit_px is not None:
        low_limit_px.columns = pd.to_datetime(low_limit_px.columns)
        low_limit_px = low_limit_px.reindex(columns=open_px.columns)
        if not bool(low_limit_px.notna().any().any()):
            low_limit_px = None

    factor_safe = None
    if factor_px is not None:
        try:
            factor_safe = factor_px.where(factor_px > 1e-9).ffill(axis=1).fillna(1.0)
        except Exception:
            factor_safe = factor_px.where(factor_px > 1e-9).fillna(1.0)
        try:
            if high_limit_px is not None:
                high_limit_px = high_limit_px * factor_safe
            if low_limit_px is not None:
                low_limit_px = low_limit_px * factor_safe
        except Exception:
            pass

    if factor_safe is not None and _want_raw(trade_price_mode):
        open_px_trade = open_px / factor_safe
        high_px_trade = high_px / factor_safe
        low_px_trade = low_px / factor_safe
        close_px_trade = close_px / factor_safe
        high_limit_px_trade = (high_limit_px / factor_safe) if high_limit_px is not None else None
        low_limit_px_trade = (low_limit_px / factor_safe) if low_limit_px is not None else None
    else:
        open_px_trade = open_px
        high_px_trade = high_px
        low_px_trade = low_px
        close_px_trade = close_px
        high_limit_px_trade = high_limit_px
        low_limit_px_trade = low_limit_px

    if factor_safe is not None and _want_raw(ref_price_mode):
        close_px_ref = close_px / factor_safe
    else:
        close_px_ref = close_px

    close_px_trade_ffill = None
    close_px_ref_ffill = None
    try:
        close_px_trade_ffill = close_px_trade.ffill(axis=1)
    except Exception:
        close_px_trade_ffill = close_px_trade
    try:
        close_px_ref_ffill = close_px_ref.ffill(axis=1)
    except Exception:
        close_px_ref_ffill = close_px_ref

    asset_list = open_px_trade.index
    dates = open_px_trade.columns
    close_prev_ref = close_px_ref_ffill.shift(1, axis=1) if close_px_ref_ffill is not None else close_px_ref.shift(1, axis=1)
    limit_pct = pd.Series(0.10, index=asset_list, dtype="float64")
    try:
        idx = limit_pct.index.astype(str)
        is_20 = idx.str.startswith("SH688") | idx.str.startswith("SZ300") | idx.str.startswith("SZ301")
        limit_pct.loc[is_20] = 0.20
    except Exception:
        limit_pct = limit_pct
    trade_mask = (dates >= trade_start) & (dates <= trade_end)
    trade_dates = dates[trade_mask]

    override_rebalance_dates = None
    if signal_df is None and pred_df is None and rebalance_n_days is None:
        if targets_override is not None or target_shares_override is not None:
            source = None
            if isinstance(target_shares_override, dict) and len(target_shares_override) > 0:
                source = target_shares_override
            if source is None and isinstance(targets_override, dict) and len(targets_override) > 0:
                source = targets_override
            if source is not None:
                try:
                    od = [pd.to_datetime(k).normalize() for k in source.keys()]
                    od = sorted(set([d for d in od if d in set(pd.DatetimeIndex(trade_dates))]))
                    if od:
                        override_rebalance_dates = pd.DatetimeIndex(od)
                except Exception:
                    override_rebalance_dates = None

    if override_rebalance_dates is not None:
        rebalance_dates = override_rebalance_dates
    elif rebalance_n_days is not None:
        try:
            n = int(rebalance_n_days)
        except Exception:
            n = None
        if n is not None and n > 0:
            rebalance_dates = pd.DatetimeIndex(trade_dates[::n])
        else:
            month_key = trade_dates.to_period("M")
            rebalance_dates = pd.Series(trade_dates, index=trade_dates).groupby(month_key).min().values
            rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    else:
        month_key = trade_dates.to_period("M")
        rebalance_dates = pd.Series(trade_dates, index=trade_dates).groupby(month_key).min().values
        rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    rebalance_set = set(rebalance_dates)
    skip_set = set(skip_months)
    date_pos = {d: i for i, d in enumerate(dates)}
    trade_pos = {d: i for i, d in enumerate(trade_dates)}

    shares = pd.Series(0.0, index=asset_list, dtype="float64")
    cash = float(account)
    prev_close_value = float(account)
    last_buy_date = pd.Series(pd.NaT, index=asset_list, dtype="datetime64[ns]")
    report_rows = []
    positions_rows = []
    trade_rows = []
    rebalance_rows = []
    if signal_df is None and pred_df is None and (targets_override is not None or target_shares_override is not None):
        signal_df = pd.DataFrame(0.0, index=pd.DatetimeIndex(dates), columns=asset_list)
    else:
        signal_df = _coerce_signal_df(signal_df=signal_df, pred_df=pred_df)
    signal_df = _align_signal_df(signal_df, dates=dates, asset_list=asset_list)
    pred_df = signal_df.stack().to_frame("score")
    pred_df.index.names = ["datetime", "instrument"]

    def _unique_keep_order(seq):
        out = []
        seen = set()
        for x in seq:
            s = str(x).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _round_px(x):
        try:
            return float(round(float(x) + 1e-12, 2))
        except Exception:
            return float(x) if x is not None else 0.0

    for d in trade_dates:
        i = date_pos.get(d)
        open_prices = open_px_trade[d]
        high_prices = high_px_trade[d] if high_px_trade is not None else None
        low_prices = low_px_trade[d] if low_px_trade is not None else None
        if close_px_trade_ffill is not None:
            close_prices = close_px_trade_ffill[d]
        else:
            close_prices = close_px_trade[d]
        open_prices = open_prices.where(open_prices > 0)
        if high_prices is not None:
            high_prices = high_prices.where(high_prices > 0)
        if low_prices is not None:
            low_prices = low_prices.where(low_prices > 0)
        close_prices = close_prices.where(close_prices > 0)
        volumes = volume_px[d] if volume_px is not None else None
        if volumes is not None:
            volumes = volumes.where(volumes > 0)
        prev_close = None
        try:
            prev_close = close_prev_ref[d]
        except Exception:
            prev_close = None

        if factor_safe is not None and prev_close is not None and _want_raw(trade_price_mode):
            try:
                tp = trade_pos.get(d, None)
                if tp is not None and tp > 0:
                    prev_d = trade_dates[tp - 1]
                    f_prev = factor_safe[prev_d]
                    f_today = factor_safe[d]
                    valid = (shares > 0) & f_prev.notna() & f_today.notna() & (f_prev > 1e-9) & (f_today > 1e-9)
                    if bool(valid.any()):
                        ratio = (f_today[valid].astype("float64") / f_prev[valid].astype("float64")).replace([np.inf, -np.inf], np.nan)
                        ratio = ratio.where(ratio > 1.005)
                        mode = str(corporate_action_mode or "cash").lower().strip()
                        if mode == "split":
                            split_ratio = ratio.fillna(1.0)
                            new_shares = (shares[valid].astype("float64") * split_ratio).replace([np.inf, -np.inf], np.nan).fillna(shares[valid].astype("float64"))
                            if lot_size and lot_size > 1:
                                new_shares = (new_shares // float(lot_size)) * float(lot_size)
                            shares.loc[valid.index] = shares.loc[valid.index].where(~valid, new_shares.astype("float64"))
                        elif mode == "none":
                            pass
                        else:
                            if mode == "auto":
                                x10 = (ratio * 10.0).round(0)
                                split_like = ratio.notna() & ((ratio - (x10 / 10.0)).abs() <= 0.002)
                                if bool(split_like.any()):
                                    split_ratio = ratio.where(split_like).fillna(1.0)
                                    new_shares = (shares[valid].astype("float64") * split_ratio).replace([np.inf, -np.inf], np.nan).fillna(shares[valid].astype("float64"))
                                    if lot_size and lot_size > 1:
                                        new_shares = (new_shares // float(lot_size)) * float(lot_size)
                                    shares.loc[valid.index] = shares.loc[valid.index].where(~valid, new_shares.astype("float64"))
                                    ratio = ratio.where(~split_like)

                            if bool(apply_dividend_cash):
                                gross_div = (prev_close[valid].astype("float64") * (1.0 - 1.0 / ratio)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                                gross_div = gross_div.clip(lower=0.0)
                                if bool((gross_div > 0).any()):
                                    held_days = (pd.to_datetime(d) - pd.to_datetime(last_buy_date[valid])).dt.days
                                    held_days = held_days.fillna(0).astype("int64")
                                    tax_rate = pd.Series(np.where(held_days < 30, 0.20, np.where(held_days < 365, 0.10, 0.0)), index=gross_div.index, dtype="float64")
                                    net_div = gross_div * (1.0 - tax_rate)
                                    div_cash = float((net_div * shares[valid].astype("float64")).sum())
                                    if div_cash != 0.0:
                                        cash += div_cash
                                        trade_rows.append(
                                            {
                                                "datetime": d,
                                                "action": "DIVIDEND",
                                                "instrument": "__DIV__",
                                                "shares": 0.0,
                                                "price": 0.0,
                                                "trade_value": 0.0,
                                                "commission": 0.0,
                                                "tax": 0.0,
                                                "slippage_rate": 0.0,
                                                "cash_after": float(cash),
                                            }
                                        )
            except Exception:
                pass
        high_limits = high_limit_px_trade[d] if high_limit_px_trade is not None else None
        low_limits = low_limit_px_trade[d] if low_limit_px_trade is not None else None
        if high_limits is None:
            try:
                high_limits = (prev_close * (1.0 + limit_pct)).round(2)
            except Exception:
                high_limits = None
        if low_limits is None:
            try:
                low_limits = (prev_close * (1.0 - limit_pct)).round(2)
            except Exception:
                low_limits = None
        if prev_close is not None:
            try:
                computed_hl = (prev_close * (1.0 + limit_pct)).round(2)
                computed_ll = (prev_close * (1.0 - limit_pct)).round(2)
                if high_limits is not None:
                    high_limits = high_limits.fillna(computed_hl)
                if low_limits is not None:
                    low_limits = low_limits.fillna(computed_ll)
            except Exception:
                high_limits = high_limits
                low_limits = low_limits

        buy_block = None
        try:
            if high_limits is not None:
                buy_block = open_prices.round(2) >= high_limits.round(2)
            elif high_prices is not None:
                buy_block = (open_prices.round(2) == high_prices.round(2)) & (close_prices.round(2) == high_prices.round(2))
        except Exception:
            buy_block = None
        sell_block = None
        try:
            if low_limits is not None:
                sell_block = open_prices.round(2) <= low_limits.round(2)
            elif low_prices is not None:
                sell_block = (open_prices.round(2) == low_prices.round(2)) & (close_prices.round(2) == low_prices.round(2))
        except Exception:
            sell_block = None

        turnover = 0.0
        total_cost = 0.0
        slip = float(slippage_rate) if slippage_rate else 0.0
        skipped = []

        if d in rebalance_set:
            if i is None:
                continue
            prev_i = i - 1 - int(signal_lag)
            targets = []
            override_target_shares = None
            force_override_exec = False
            skip_rebalance_trade = False

            override_targets = None
            if targets_override is not None:
                try:
                    override_targets = targets_override.get(pd.to_datetime(d).normalize())
                except Exception:
                    override_targets = None
            if target_shares_override is not None:
                try:
                    override_target_shares = target_shares_override.get(pd.to_datetime(d).normalize())
                except Exception:
                    override_target_shares = None
            if override_targets is not None:
                if isinstance(override_targets, str):
                    override_targets = [x.strip() for x in override_targets.split(",") if str(x).strip()]
                try:
                    targets = _unique_keep_order(list(override_targets))
                except Exception:
                    targets = []
            elif override_target_shares is not None:
                if isinstance(override_target_shares, pd.Series):
                    override_target_shares = override_target_shares.to_dict()
                if isinstance(override_target_shares, pd.DataFrame):
                    if override_target_shares.shape[1] == 1:
                        override_target_shares = override_target_shares.iloc[:, 0].to_dict()
                    else:
                        override_target_shares = dict(override_target_shares.to_dict(orient="records"))
                if isinstance(override_target_shares, dict):
                    try:
                        override_target_shares = {str(k): float(v) for k, v in override_target_shares.items()}
                    except Exception:
                        override_target_shares = None
                else:
                    override_target_shares = None
                if override_target_shares is not None:
                    targets = _unique_keep_order(list(override_target_shares.keys()))
                    force_override_exec = True
            elif prev_i >= 0:
                prev_d = dates[prev_i]
                scores = signal_df.loc[prev_d]
                pool = _pool_on(prev_d)
                if pool is not None:
                    try:
                        scores = scores.reindex(pool).fillna(0.0)
                    except Exception:
                        scores = scores
                ranked_index = _rank_targets(scores, topk=10**9, pool_order=pool)
                if ranked_index:
                    initial_targets = ranked_index[: int(topk)]
                    if not bool(backfill_untradable):
                        targets = [str(s) for s in initial_targets]
                    else:
                        selectable = open_prices.notna() & (open_prices > 0)
                        if volumes is not None:
                            selectable = selectable & volumes.notna() & (volumes > 0)
                        if high_limits is not None:
                            selectable = selectable & (open_prices < (high_limits - 1e-9))
                        if buy_block is not None:
                            selectable = selectable & (~buy_block)
                        targets = []
                        for s in ranked_index:
                            if bool(selectable.get(s, False)):
                                targets.append(str(s))
                                if len(targets) >= int(topk):
                                    break
                            else:
                                if len(targets) < int(topk):
                                    skipped.append(str(s))

            if d.month in skip_set:
                skip_rebalance_trade = True

            if not targets:
                skip_rebalance_trade = True

            if (not bool(skip_rebalance_trade)) and targets and override_target_shares is None and target_value_per_stock is None:
                try:
                    mark_px = open_prices.copy()
                    if prev_close is not None:
                        mark_px = mark_px.fillna(prev_close)
                    valid_mark = mark_px.notna() & (mark_px > 0)
                    port_value = float(cash) + float((shares[valid_mark] * mark_px[valid_mark]).sum())
                except Exception:
                    port_value = float(cash)
                if lot_size and lot_size > 1 and port_value > 0:
                    final_k = 0
                    for k in range(len(targets), 0, -1):
                        budget_k = float(port_value) / float(k)
                        ok = True
                        for s in targets[:k]:
                            px = float(open_prices.get(s, 0.0))
                            if not (px > 0) and prev_close is not None:
                                try:
                                    px = float(prev_close.get(s, 0.0))
                                except Exception:
                                    px = 0.0
                            if px <= 0.0 or budget_k / px < float(lot_size):
                                ok = False
                                break
                        if ok:
                            final_k = int(k)
                            break
                    if final_k <= 0:
                        skip_rebalance_trade = True
                    else:
                        targets = list(targets[:final_k])

            if bool(skip_rebalance_trade):
                holdings_after = shares[shares > 0].index.tolist()
                rebalance_rows.append(
                    {
                        "datetime": d,
                        "targets": ",".join(list(map(str, targets))) if targets else "",
                        "holdings_after": ",".join(list(map(str, holdings_after))) if holdings_after else "",
                        "cash_after": float(cash),
                        "skipped": ",".join(list(map(str, skipped))) if skipped else "",
                    }
                )
            else:
                current_holdings = shares[shares > 0].index.tolist()
                sell_list = [s for s in current_holdings if s not in targets]
                for s in sell_list:
                    px = float(open_prices.get(s, 0.0))
                    if not (px > 0):
                        if prev_close is not None:
                            try:
                                px = float(prev_close.get(s, 0.0))
                            except Exception:
                                px = 0.0
                        if not (px > 0):
                            continue
                    if not bool(force_override_exec):
                        if volumes is not None and not (float(volumes.get(s, 0.0)) > 0):
                            continue
                        if sell_block is not None and bool(sell_block.get(s, False)):
                            continue
                        if low_limits is not None:
                            ll = low_limits.get(s, None)
                            if ll is not None and pd.notna(ll):
                                if round(px, 2) <= round(float(ll), 2):
                                    continue
                    cur_sh = float(shares.get(s, 0.0))
                    if not (cur_sh > 0):
                        continue
                    sell_shares = cur_sh
                    if lot_size and lot_size > 1:
                        sell_shares = (sell_shares // float(lot_size)) * float(lot_size)
                    if not (sell_shares > 0):
                        continue
                    exec_px = px * (1.0 - slip) if slip > 0 else px
                    exec_px = _round_px(exec_px)
                    ref_px = None
                    if prev_close is not None:
                        try:
                            ref_px = prev_close.get(s, None)
                        except Exception:
                            ref_px = None
                    if ref_px is None or (not pd.notna(ref_px)) or (not (float(ref_px) > 0)):
                        ref_px = float(px)
                    if sell_limit_multiplier is not None:
                        sell_limit_px = _round_px(float(ref_px) * float(sell_limit_multiplier))
                        if exec_px < sell_limit_px - 1e-9:
                            continue
                    trade_value = sell_shares * exec_px
                    comm = max(float(min_commission), trade_value * float(commission_rate)) if commission_rate and trade_value > 0 else 0.0
                    tax = trade_value * float(stamp_tax_rate) if stamp_tax_rate else 0.0
                    cost = comm + tax
                    cash += trade_value - cost
                    turnover += trade_value
                    total_cost += cost
                    shares.loc[s] = max(0.0, cur_sh - sell_shares)
                    trade_rows.append(
                        {
                            "datetime": d,
                            "action": "SELL",
                            "instrument": s,
                            "shares": float(sell_shares),
                            "price": float(exec_px),
                            "trade_value": float(trade_value),
                            "commission": float(comm),
                            "tax": float(tax),
                            "slippage_rate": float(slip),
                            "cash_after": float(cash),
                        }
                    )

            if targets:
                min_trade_shares = float(lot_size) if lot_size else 0.0
                target_shares_map = {}
                if override_target_shares is None:
                    for s in targets:
                        px_open = float(open_prices.get(s, 0.0))
                        if not (px_open > 0):
                            target_shares_map[str(s)] = 0.0
                            continue
                        ref_px = None
                        if prev_close is not None:
                            try:
                                ref_px = prev_close.get(s, None)
                            except Exception:
                                ref_px = None
                        if ref_px is None or (not pd.notna(ref_px)) or (not (float(ref_px) > 0)):
                            ref_px = float(px_open)
                        if target_value_per_stock is None:
                            try:
                                mark_px = open_prices.copy()
                                if prev_close is not None:
                                    mark_px = mark_px.fillna(prev_close)
                                valid_mark = mark_px.notna() & (mark_px > 0)
                                port_value = float(cash) + float((shares[valid_mark] * mark_px[valid_mark]).sum())
                            except Exception:
                                port_value = float(cash)
                            per_value = float(port_value) / float(len(targets)) if targets else 0.0
                        else:
                            per_value = float(target_value_per_stock)
                        desired = float(per_value) / float(px_open) if float(px_open) > 0 else 0.0
                        if lot_size and lot_size > 1:
                            desired = (desired // float(lot_size)) * float(lot_size)
                        desired = float(max(desired, 0.0))
                        if min_trade_shares > 0 and desired < min_trade_shares:
                            desired = 0.0
                        target_shares_map[str(s)] = desired

                def _max_affordable_shares(cash_amount, trade_price, desired_shares):
                    try:
                        cash_amount = float(cash_amount)
                        trade_price = float(trade_price)
                        desired_shares = float(desired_shares)
                    except Exception:
                        return 0.0
                    if not (cash_amount > 0) or not (trade_price > 0) or not (desired_shares > 0):
                        return 0.0
                    lot = int(lot_size) if lot_size and lot_size > 1 else 1
                    max_lots = int(desired_shares // lot)
                    if max_lots <= 0:
                        return 0.0

                    def _spend(lots):
                        sh = float(lots * lot)
                        tv = sh * trade_price
                        cm = max(float(min_commission), tv * float(commission_rate)) if commission_rate and tv > 0 else 0.0
                        return float(tv + cm)

                    lo, hi = 0, max_lots
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if _spend(mid) <= cash_amount + 1e-9:
                            lo = mid
                        else:
                            hi = mid - 1
                    return float(lo * lot)

                exec_px_map = {}
                cur_sh_map = {}
                tgt_sh_map = {}
                delta_map = {}
                tradable_map = {}
                for s in targets:
                    exec_px = float(open_prices.get(s, 0.0))
                    if not (exec_px > 0) and bool(force_override_exec):
                        if prev_close is not None:
                            try:
                                exec_px = float(prev_close.get(s, 0.0))
                            except Exception:
                                exec_px = exec_px
                    if not (exec_px > 0):
                        tradable_map[str(s)] = False
                        continue
                    if not bool(force_override_exec):
                        if volumes is not None and not (float(volumes.get(s, 0.0)) > 0):
                            tradable_map[str(s)] = False
                            continue
                        if buy_block is not None and bool(buy_block.get(s, False)):
                            tradable_map[str(s)] = False
                            continue
                        if sell_block is not None and bool(sell_block.get(s, False)):
                            tradable_map[str(s)] = False
                            continue
                    tradable_map[str(s)] = True
                    exec_px_map[str(s)] = float(exec_px)
                    cur_sh = float(shares.get(s, 0.0))
                    cur_sh_map[str(s)] = cur_sh
                    if override_target_shares is not None:
                        tgt = float(override_target_shares.get(str(s), 0.0))
                    else:
                        tgt = float(target_shares_map.get(str(s), 0.0))
                    if lot_size and lot_size > 1:
                        tgt = (tgt // float(lot_size)) * float(lot_size)
                    if tgt < 0:
                        tgt = 0.0
                    tgt_sh_map[str(s)] = float(tgt)
                    delta_map[str(s)] = float(tgt - cur_sh)

                for s in targets:
                    s = str(s)
                    if not bool(tradable_map.get(s, False)):
                        continue
                    exec_px = float(exec_px_map.get(s, 0.0))
                    cur_sh = float(cur_sh_map.get(s, float(shares.get(s, 0.0))))
                    tgt = float(tgt_sh_map.get(s, 0.0))
                    delta = float(delta_map.get(s, tgt - cur_sh))
                    if min_trade_shares > 0 and abs(delta) < min_trade_shares and tgt != 0:
                        continue
                    if not (delta < 0):
                        continue
                    if not bool(force_override_exec):
                        if low_limits is not None:
                            ll = low_limits.get(s, None)
                            if ll is not None and pd.notna(ll):
                                if round(exec_px, 2) <= round(float(ll), 2):
                                    continue
                    sell_shares = -delta
                    if lot_size and lot_size > 1:
                        sell_shares = (sell_shares // float(lot_size)) * float(lot_size)
                    sell_shares = min(sell_shares, cur_sh)
                    if not (sell_shares > 0):
                        continue
                    trade_px = exec_px * (1.0 - slip) if slip > 0 else exec_px
                    trade_px = _round_px(trade_px)
                    ref_px = None
                    if prev_close is not None:
                        try:
                            ref_px = prev_close.get(s, None)
                        except Exception:
                            ref_px = None
                    if ref_px is None or (not pd.notna(ref_px)) or (not (float(ref_px) > 0)):
                        ref_px = float(exec_px)
                    if sell_limit_multiplier is not None:
                        sell_limit_px = _round_px(float(ref_px) * float(sell_limit_multiplier))
                        if trade_px < sell_limit_px - 1e-9:
                            continue
                    trade_value = sell_shares * trade_px
                    comm = max(float(min_commission), trade_value * float(commission_rate)) if commission_rate and trade_value > 0 else 0.0
                    tax = trade_value * float(stamp_tax_rate) if stamp_tax_rate else 0.0
                    cost = comm + tax
                    cash += trade_value - cost
                    turnover += trade_value
                    total_cost += cost
                    shares.loc[s] = max(0.0, cur_sh - sell_shares)
                    cur_sh_map[s] = float(shares.get(s, 0.0))
                    trade_rows.append(
                        {
                            "datetime": d,
                            "action": "SELL",
                            "instrument": s,
                            "shares": float(sell_shares),
                            "price": float(trade_px),
                            "trade_value": float(trade_value),
                            "commission": float(comm),
                            "tax": float(tax),
                            "slippage_rate": float(slip),
                            "cash_after": float(cash),
                        }
                    )

                for s in targets:
                    s = str(s)
                    if not bool(tradable_map.get(s, False)):
                        continue
                    exec_px = float(exec_px_map.get(s, 0.0))
                    cur_sh = float(shares.get(s, 0.0))
                    tgt = float(tgt_sh_map.get(s, 0.0))
                    delta = float(tgt - cur_sh)
                    if min_trade_shares > 0 and abs(delta) < min_trade_shares and tgt != 0:
                        continue
                    if not (delta > 0):
                        continue
                    if not bool(force_override_exec):
                        if high_limits is not None:
                            hl = high_limits.get(s, None)
                            if hl is not None and pd.notna(hl):
                                if round(exec_px, 2) >= round(float(hl), 2):
                                    continue
                    buy_shares = delta
                    if lot_size and lot_size > 1:
                        buy_shares = (buy_shares // float(lot_size)) * float(lot_size)
                    if not (buy_shares > 0):
                        continue
                    trade_px = exec_px * (1.0 + slip) if slip > 0 else exec_px
                    trade_px = _round_px(trade_px)
                    ref_px = None
                    if prev_close is not None:
                        try:
                            ref_px = prev_close.get(s, None)
                        except Exception:
                            ref_px = None
                    if ref_px is None or (not pd.notna(ref_px)) or (not (float(ref_px) > 0)):
                        ref_px = float(exec_px)
                    if buy_limit_multiplier is not None:
                        buy_limit_px = _round_px(float(ref_px) * float(buy_limit_multiplier))
                        if trade_px > buy_limit_px + 1e-9:
                            continue

                    def _spend_for(sh):
                        tv = float(sh) * float(trade_px)
                        cm = max(float(min_commission), tv * float(commission_rate)) if commission_rate and tv > 0 else 0.0
                        return float(tv + cm), float(tv), float(cm)

                    spend, trade_value, comm = _spend_for(buy_shares)
                    if spend > cash:
                        if lot_size and lot_size > 1:
                            max_aff = _max_affordable_shares(cash, trade_px, buy_shares)
                            if max_aff > 0:
                                buy_shares = max_aff
                                spend, trade_value, comm = _spend_for(buy_shares)
                        if spend > cash:
                            continue
                    cash -= spend
                    turnover += trade_value
                    total_cost += comm
                    shares.loc[s] = cur_sh + buy_shares
                    last_buy_date.loc[s] = pd.to_datetime(d)
                    trade_rows.append(
                        {
                            "datetime": d,
                            "action": "BUY",
                            "instrument": s,
                            "shares": float(buy_shares),
                            "price": float(trade_px),
                            "trade_value": float(trade_value),
                            "commission": float(comm),
                            "tax": 0.0,
                            "slippage_rate": float(slip),
                            "cash_after": float(cash),
                        }
                    )

                cash = float(max(cash, 0.0))
                holdings_after = shares[shares > 0].index.tolist()
                rebalance_rows.append(
                    {
                        "datetime": d,
                        "targets": ",".join(list(map(str, targets))) if targets else "",
                        "holdings_after": ",".join(list(map(str, holdings_after))) if holdings_after else "",
                        "cash_after": float(cash),
                        "skipped": ",".join(list(map(str, skipped))) if skipped else "",
                    }
                )

        mark_close = close_prices
        try:
            if prev_close is not None:
                mark_close = mark_close.fillna(prev_close)
        except Exception:
            mark_close = mark_close
        valid_close = mark_close.notna()
        close_value = cash + float((shares[valid_close] * mark_close[valid_close]).sum())
        daily_ret = (close_value / prev_close_value - 1.0) if prev_close_value > 0 else 0.0
        prev_close_value = close_value

        report_rows.append(
            {
                "datetime": d,
                "account": close_value,
                "return": daily_ret,
                "total_turnover": turnover,
                "total_cost": total_cost,
                "cash": cash,
                "pos": int((shares > 0).sum()),
            }
        )
        positions_rows.append(pd.Series(shares.to_numpy(copy=True), index=asset_list, name=d))

    report_df = pd.DataFrame(report_rows).set_index("datetime")
    positions_df = pd.DataFrame(positions_rows)
    positions_df.index = pd.to_datetime(positions_df.index)

    bench_ret = pd.Series(0.0, index=report_df.index, dtype="float64")
    try:
        first_day = pd.to_datetime(report_df.index.min())
        load_start = (first_day - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        bench_fields = ['$close']
        if _want_factor(benchmark_price_mode) or _want_raw(benchmark_price_mode):
            bench_fields.append('$factor')
        bench_df = QlibData.features([benchmark], bench_fields, start_time=load_start, end_time=end_time)
        if bench_df.index.names == ['instrument', 'datetime']:
            bench_close = bench_df['$close'].unstack(level='datetime').iloc[0]
            bench_factor = bench_df['$factor'].unstack(level='datetime').iloc[0] if '$factor' in bench_df.columns else None
        else:
            bench_close = bench_df['$close'].unstack(level='instrument').iloc[:, 0]
            bench_factor = bench_df['$factor'].unstack(level='instrument').iloc[:, 0] if '$factor' in bench_df.columns else None
        bench_close.index = pd.to_datetime(bench_close.index)
        if bench_factor is not None and _want_raw(benchmark_price_mode):
            bench_factor.index = pd.to_datetime(bench_factor.index)
            bench_factor = bench_factor.where(bench_factor > 1e-9).fillna(1.0)
            bench_close = bench_close / bench_factor
        elif bench_factor is not None and _want_factor(benchmark_price_mode):
            bench_factor.index = pd.to_datetime(bench_factor.index)
            bench_factor = bench_factor.where(bench_factor > 1e-9).fillna(1.0)
            bench_close = bench_close * bench_factor
        prev_days = bench_close.index[bench_close.index < first_day]
        if len(prev_days) > 0:
            prev_day = prev_days.max()
        else:
            prev_day = first_day
        full_index = pd.DatetimeIndex([prev_day]).append(pd.to_datetime(report_df.index))
        bench_close = bench_close.reindex(full_index).ffill()
        bench_ret = bench_close.pct_change().reindex(pd.to_datetime(report_df.index)).fillna(0.0)
    except Exception:
        bench_ret = bench_ret

    report_df["bench"] = bench_ret
    analysis = risk_analysis(report_df["return"] - report_df["bench"])
    abs_analysis = risk_analysis(report_df["return"])
    trades_df = pd.DataFrame(trade_rows)
    rebalances_df = pd.DataFrame(rebalance_rows).set_index("datetime") if rebalance_rows else pd.DataFrame(columns=["targets", "holdings_after", "cash_after"]).set_index(pd.DatetimeIndex([], name="datetime"))
    if bool(return_trades) and bool(return_rebalances):
        return report_df, positions_df, pred_df, analysis, abs_analysis, trades_df, rebalances_df
    if bool(return_trades):
        return report_df, positions_df, pred_df, analysis, abs_analysis, trades_df
    if bool(return_rebalances):
        return report_df, positions_df, pred_df, analysis, abs_analysis, rebalances_df
    return report_df, positions_df, pred_df, analysis, abs_analysis


def backtest_summary_row(label, analysis, report_df=None):
    total_ret = 0.0
    bench_ret = 0.0

    if report_df is not None and isinstance(report_df, pd.DataFrame):
        if "return" in report_df.columns:
            ret_s = report_df["return"].dropna()
            total_ret = float(ret_s.add(1.0).prod() - 1.0) if len(ret_s) else 0.0
        if "bench" in report_df.columns:
            bench_s = report_df["bench"].dropna()
            bench_ret = float(bench_s.add(1.0).prod() - 1.0) if len(bench_s) else 0.0

    excess_ret = float(total_ret - bench_ret)
    ir = float(analysis.loc["information_ratio", "risk"])
    mdd = float(analysis.loc["max_drawdown", "risk"])

    return {
        "label": label,
        "TotalReturn%": float(total_ret * 100.0),
        "BenchReturn%": float(bench_ret * 100.0),
        "ExcessReturn%": float(excess_ret * 100.0),
        "MaxDD%": float(mdd * 100.0),
        "IR": ir,
    }
