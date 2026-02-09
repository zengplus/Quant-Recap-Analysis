import pandas as pd
import torch
import os
import sys
from .qlib_loader import QlibDataLoader
from .vm import StackVM

def compute_signal_matrix(
    formula,
    strategy_path=None,
    provider_uri=None,
    instruments='csi300',
    start_time=None,
    end_time=None,
    lookback_days=730,
    token_mode="postfix",
    price_mode="pre_adjusted",
    monthly_resample=True,
):
    """
    Computes the signal matrix for the given formula and period.
    Returns:
        signal_df: DataFrame with index (date, instrument) and columns ['score'] (or just signals)
        analysis: Some analysis result (optional, returning None for now)
    """
    
    # 1. Load Data
    # We need to load data with buffer for rolling windows
    load_start = pd.to_datetime(start_time) - pd.Timedelta(days=lookback_days)
    load_start_str = load_start.strftime("%Y-%m-%d")
    
    loader = QlibDataLoader(provider_uri=provider_uri)
    loader.load_data(
        start_time=load_start_str, 
        end_time=end_time, 
        instruments=instruments, 
        verbose=False, 
        price_mode=price_mode
    )
    
    # 2. Slice to the requested period (after buffer)
    # Similar logic to run_pipeline.py's slice_loader but we keep the data for calculation
    # Actually, StackVM needs the full tensor to compute rolling windows correctly.
    # We execute on the full tensor, THEN slice the result.
    
    vm = StackVM()
    try:
        formula_tokens = [int(t) for t in list(formula)]
    except Exception:
        formula_tokens = [0]
    if token_mode == "prefix":
        formula_tokens = vm.repair_prefix(formula_tokens)
    else:
        formula_tokens = vm.repair_postfix(formula_tokens)
    
    # Execute formula
    if token_mode == "prefix":
        res = vm.execute_prefix(formula_tokens, loader.feat_tensor)
    else:
        res = vm.execute(formula_tokens, loader.feat_tensor)
        
    if res is None:
        # Try repair
        if token_mode == "prefix":
            repaired = vm.repair_prefix(formula_tokens)
            res = vm.execute_prefix(repaired, loader.feat_tensor)
        else:
            repaired = vm.repair_postfix(formula_tokens)
            res = vm.execute(repaired, loader.feat_tensor)
            
    if res is None:
        # Fallback to zeros
        res = torch.zeros((loader.feat_tensor.shape[0], loader.feat_tensor.shape[2]), device=loader.feat_tensor.device)
        
    # res is (Assets, Time)
    # We need to convert to DataFrame
    
    # Slice to requested period
    dates = loader.dates # Index of dates
    assets = loader.stock_list # List of assets
    
    # Find start index
    start_dt = pd.Timestamp(start_time)
    start_idx = dates.searchsorted(start_dt)
    
    # Slice result and dates
    res_sliced = res[:, start_idx:]
    dates_sliced = dates[start_idx:]
    
    # Convert to DataFrame
    # Rows: Dates, Cols: Assets
    # But usually signal_df is (Dates, Assets) or Stacked.
    # run_joinquant_backtest expects signal_df indexed by datetime or (datetime, instrument).
    
    res_np = res_sliced.cpu().numpy().T # (Time, Assets)
    
    signal_df = pd.DataFrame(res_np, index=dates_sliced, columns=assets)
    
    if monthly_resample:
        # Resample to monthly (end of month or first day?)
        # Usually we just take the signal at the rebalance date.
        # But for now, we return daily signals and let the backtester handle rebalancing.
        # If monthly_resample is True, maybe we forward fill or something.
        # For simplicity, we return daily signals.
        pass
        
    return signal_df, None
