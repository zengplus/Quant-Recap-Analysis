""" 
File: engine.py
可配置版本：所有参数集中管理
"""
import sys
import os

# Add project root to sys.path to allow running as script
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "model_core"

import torch
import gc
from torch.distributions import Categorical
from tqdm import tqdm
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
# from concurrent.futures import ThreadPoolExecutor
from qlib.data import D
from qlib.backtest import backtest as qlib_backtest
from qlib.contrib.strategy import TopkDropoutStrategy

from .config import ModelConfig
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .qlib_loader import QlibDataLoader
from .ashare_backtest import AShareBacktest
from .market_state_eval import build_pred_df_from_factor_scores, evaluate_state_metrics, extract_index_close

# ============== 配置类 ==============
@dataclass
class TrainingConfig:
    """训练配置参数"""
    # 基础训练参数
    batch_size: int = 256
    train_steps: int = 1000
    max_formula_len: int = 20
    learning_rate: float = 1e-3
    entropy_coef: float = 0.01
    grad_clip_norm: float = 1.0
    
    # 数据参数
    train_start: str = '2021-01-01'
    train_end: str = '2023-12-31'
    valid_start: str = '2024-01-01'
    valid_end: str = '2024-12-31'
    test_start: str = '2025-01-01'
    test_end: str = '2025-12-31'
    instruments: str = 'csi300'
    
    # LoRD正则化参数
    use_lord: bool = True
    lord_decay_rate: float = 1e-3
    lord_num_iterations: int = 5
    rank_monitor_interval: int = 10
    
    # 奖励函数参数
    reward_metric: str = "hybrid" # ic, backtest, hybrid
    ic_reward_weight: float = 1.0
    backtest_reward_weight: float = 0.05
    use_turnover_penalty: bool = True
    turnover_penalty_coef: float = 0.35
    syntax_error_penalty: float = -1.0
    constant_penalty: float = -0.2
    eval_error_penalty: float = -0.5
    base_reward: float = 0.0
    positive_ic_multiplier: float = 5.0
    negative_ic_multiplier: float = 0.3
    ic_clip: float = 0.5  # IC值截断范围 [-0.5, 0.5]
    
    # Top-K策略参数
    topk: int = 10
    rebalance_period: int = 1
    
    # 换手率硬性约束
    use_turnover_hard_constraint: bool = False
    max_turnover_threshold: float = 10.0  # 对应年化1000%
    turnover_breach_penalty: float = -2.0 # 超出阈值后的惩罚分数
    
    # 验证参数
    validation_interval: int = 200
    validation_top_k: int = 1
    train_sample_assets: Optional[int] = 64
    train_sample_time: Optional[int] = 128
    valid_sample_assets: Optional[int] = 128
    valid_sample_time: Optional[int] = 128
    ic_sample_assets: Optional[int] = None
    ic_time_stride: int = 1
    eval_cache_interval: int = 20
    eval_cache_clear_interval: int = 100
    gc_interval: int = 100
    early_stop_patience: int = 50  # 早停耐心值
    early_stop_min_delta: float = 0.001
    early_stop_start_ratio: float = 0.9
    
    # 保存参数
    save_checkpoint_interval: int = 200
    checkpoint_history_max_len: int = 0
    output_dir: str = "outputs"
    save_checkpoints: bool = True
    save_config: bool = True
    save_best_strategy: bool = True
    save_model_weights: bool = True
    save_training_history: bool = True
    plot_training_history: bool = False
    
    # 调试参数
    debug_mode: bool = False
    test_simple_formulas: bool = True
    log_interval: int = 10

    def to_dict(self):
        return self.__dict__

# ============== 训练引擎 ==============
class ConfigurableAShareAlphaEngine:
    """
    可配置版本训练引擎
    """
    def __init__(self, config: TrainingConfig, 
                 train_loader=None, valid_loader=None, test_loader=None):
        # 保存配置
        self.config = config
        
        # 数据加载器（可通过参数传入或内部创建）
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        # 确保输出目录存在
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
        
        ModelConfig.MAX_FORMULA_LEN = config.max_formula_len

        # 1. 初始化模型
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        print(f"🚀 Using Device: {ModelConfig.DEVICE}")
        
        # 2. 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        # 3. LoRD 正则化器
        self.lord_opt = None
        self.rank_monitor = None
        if config.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=config.lord_decay_rate,
                num_iterations=config.lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
            )
        
        # 4. 评估组件
        self.vm = StackVM()
        # self.backtest = AShareBacktest(
        #     topk=config.topk, 
        #     rebalance_period=config.rebalance_period
        # )
        
        # 5. 训练状态
        self.best_valid_score = -float('inf')
        self.best_formula = None
        self.training_history = {
            'step': [],
            'train_reward': [],
            'train_reward_std': [],
            'valid_score': [],
            'stable_rank': []
        }
        self._train_eval_cache = None
        self._valid_eval_cache = None
        self._train_cache_step = -1
        self._valid_cache_step = -1
        
        # 6. 早停器
        self.early_stopper = EarlyStopper(
            patience=config.early_stop_patience,
            min_delta=config.early_stop_min_delta
        )
        
        # 7. 预计算token属性
        self._init_token_masking()
        
        # ThreadPoolExecutor is REMOVED because it causes massive overhead on MPS
        # max_workers = min(32, (os.cpu_count() or 1) * 4)
        # self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 8. 测试简单公式（可选）
        if config.test_simple_formulas and self.train_loader:
            self._test_simple_formulas()
    
    def _init_token_masking(self):
        """初始化token masking属性"""
        self.vocab_size = self.model.vocab_size
        self.token_change = torch.ones(self.vocab_size, dtype=torch.long, 
                                      device=ModelConfig.DEVICE)
        self.token_arity = torch.zeros(self.vocab_size, dtype=torch.long, 
                                      device=ModelConfig.DEVICE)
        
        # 从vm获取操作符信息
        for token_id, arity in self.vm.arity_map.items():
            if token_id < self.vocab_size:
                self.token_change[token_id] = 1 - arity
                self.token_arity[token_id] = arity
    
    def _test_simple_formulas(self):
        """测试简单公式确保评估系统正常工作"""
        print("🧪 简单公式测试:")
        test_formulas = [
            [0],  # 单个特征
            [0, 1, self.vm.feat_offset],  # 特征0 + 特征1
        ]
        
        results = []
        for i, formula in enumerate(test_formulas):
            train_score = self.evaluate_formula(formula, self.train_loader)
            
            result = {
                'formula': str(formula),
                'train_score': train_score
            }
            
            if self.valid_loader:
                valid_score = self.evaluate_formula(formula, self.valid_loader)
                result['valid_score'] = valid_score
            
            results.append(result)
        
        # 保存测试结果
        test_df = pd.DataFrame(results)
        test_df.to_csv(f"{self.config.output_dir}/simple_formulas_test.csv", index=False)
        
        # 打印结果
        for idx, row in test_df.iterrows():
            if 'valid_score' in row:
                print(f"  公式{idx}: {row['formula']} -> 训练集: {row['train_score']:.4f}, 验证集: {row['valid_score']:.4f}")
            else:
                print(f"  公式{idx}: {row['formula']} -> 训练集: {row['train_score']:.4f}")
        print()
    
    def evaluate_formula(self, formula_tokens, data_loader):
        """评估公式表现"""
        return self._evaluate_formula_with_tensors(
            formula_tokens,
            data_loader.feat_tensor,
            data_loader.raw_data_cache,
            data_loader.target_ret
        )

    def _clear_eval_cache(self, cache_name=None):
        if cache_name is None or cache_name == "train":
            self._train_eval_cache = None
            self._train_cache_step = -1
        if cache_name is None or cache_name == "valid":
            self._valid_eval_cache = None
            self._valid_cache_step = -1

    def _get_eval_tensors(self, data_loader, sample_assets, sample_time, cache_name="train", step=0):
        if cache_name == "train":
            cache = self._train_eval_cache
            cache_step = self._train_cache_step
        else:
            cache = self._valid_eval_cache
            cache_step = self._valid_cache_step

        use_cache = self.config.eval_cache_interval > 0
        if use_cache and cache and cache["assets"] == sample_assets and cache["time"] == sample_time:
            if step - cache_step < self.config.eval_cache_interval:
                return cache["feat"], cache["raw"], cache["ret"]
            self._clear_eval_cache(cache_name)

        feat_tensor = data_loader.feat_tensor
        raw_data_cache = data_loader.raw_data_cache
        target_ret = data_loader.target_ret

        if sample_assets and feat_tensor.shape[0] > sample_assets:
            idx_assets = torch.randperm(feat_tensor.shape[0], device=feat_tensor.device)[:sample_assets]
            feat_tensor = feat_tensor.index_select(0, idx_assets)
            raw_data_cache = {k: v.index_select(0, idx_assets) for k, v in raw_data_cache.items()}
            target_ret = target_ret.index_select(0, idx_assets)

        if sample_time and feat_tensor.shape[2] > sample_time:
            start = int(torch.randint(0, feat_tensor.shape[2] - sample_time + 1, (1,)).item())
            end = start + sample_time
            feat_tensor = feat_tensor[:, :, start:end]
            raw_data_cache = {k: v[:, start:end] for k, v in raw_data_cache.items()}
            target_ret = target_ret[:, start:end]

        if use_cache:
            if cache_name == "train":
                self._train_eval_cache = {
                    "assets": sample_assets,
                    "time": sample_time,
                    "feat": feat_tensor,
                    "raw": raw_data_cache,
                    "ret": target_ret
                }
                self._train_cache_step = step
            else:
                self._valid_eval_cache = {
                    "assets": sample_assets,
                    "time": sample_time,
                    "feat": feat_tensor,
                    "raw": raw_data_cache,
                    "ret": target_ret
                }
                self._valid_cache_step = step

        return feat_tensor, raw_data_cache, target_ret

    def _evaluate_formula_with_tensors(self, formula_tokens, feat_tensor, raw_data_cache, target_ret):
        with torch.no_grad():
            try:
                result = self.vm.execute(formula_tokens, feat_tensor)
                if result is None:
                    return self.config.syntax_error_penalty
                
                result_std = result.std().item()
                if result_std < 1e-6:
                    return self.config.constant_penalty
                
                if torch.isnan(result).any() or torch.isinf(result).any():
                    return self.config.constant_penalty
                
                result_mean = result.mean().item()
                result = (result - result_mean) / (result_std + 1e-8)
                
                try:
                    # --------------------------------------------------------------------------------
                    # 核心改造：使用 IC (Information Coefficient) 作为奖励信号
                    # IC衡量了因子预测值与未来收益的线性相关性，计算速度快，适合在训练中作为奖励
                    # --------------------------------------------------------------------------------
                    
                    # 展平因子和收益张量
                    pred = result.reshape(-1)
                    target = target_ret.reshape(-1)

                    # 移除NaN值
                    valid_mask = ~torch.isnan(pred) & ~torch.isnan(target)
                    if valid_mask.sum() < 2:
                        return self.config.eval_error_penalty

                    pred = pred[valid_mask]
                    target = target[valid_mask]
                    
                    # 计算皮尔逊相关系数 (IC)
                    pred_mean = pred.mean()
                    target_mean = target.mean()
                    
                    pred_centered = pred - pred_mean
                    target_centered = target - target_mean
                    
                    numerator = (pred_centered * target_centered).sum()
                    denominator = torch.sqrt((pred_centered**2).sum() * (target_centered**2).sum())
                    
                    if denominator < 1e-8:
                        ic_val = 0.0
                    else:
                        ic_val = (numerator / denominator).item()
                        
                    # 应用奖励塑形
                    ic_clip = self.config.ic_clip
                    if ic_clip is not None:
                        ic_val = max(min(ic_val, ic_clip), -ic_clip)

                    if ic_val >= 0:
                        reward = self.config.base_reward + ic_val * self.config.positive_ic_multiplier
                    else:
                        reward = self.config.base_reward + ic_val * self.config.negative_ic_multiplier
                    
                    return reward
                    
                except Exception as e:
                    if self.config.debug_mode:
                        print(f"IC 计算异常: {e}")
                    return self.config.eval_error_penalty
                    
            except Exception as e:
                if self.config.debug_mode:
                    print(f"公式评估异常: {e}")
                return self.config.syntax_error_penalty
    
    def train(self):
        """主训练循环"""
        print("🚀 Starting Alpha Mining (Configurable Version)...")
        print(f"📊 配置参数:")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Train Steps: {self.config.train_steps}")
        print(f"  Formula Length: {self.config.max_formula_len}")
        print(f"  Learning Rate: {self.config.learning_rate}")
        print(f"  Entropy Coef: {self.config.entropy_coef}")
        
        if self.config.use_lord:
            print(f"  LoRD Decay Rate: {self.config.lord_decay_rate}")
        
        print("-" * 50)
        
        pbar = tqdm(range(self.config.train_steps), mininterval=0.1, smoothing=0.0)
        early_stop_start = int(self.config.train_steps * self.config.early_stop_start_ratio)
        last_valid_score = np.nan
        
        for step in pbar:
            # 1. 生成公式
            formulas, log_probs, entropies = self._generate_formulas_batch()
            formulas_list = formulas.detach().cpu().tolist()
            
            # 2. 评估训练集
            train_scores = self._evaluate_formulas_batch(
                formulas_list,
                self.train_loader,
                self.config.train_sample_assets,
                self.config.train_sample_time,
                step
            )
            
            # 3. 验证集评估（按间隔或每步）
            valid_score = -999
            if self.valid_loader and (step % self.config.validation_interval == 0 or step == self.config.train_steps - 1):
                top_k = max(1, min(self.config.validation_top_k, self.config.batch_size))
                top_indices = torch.topk(train_scores, k=top_k).indices.detach().cpu().tolist()
                valid_scores = []
                feat_tensor, raw_data_cache, target_ret = self._get_eval_tensors(
                    self.valid_loader,
                    self.config.valid_sample_assets,
                    self.config.valid_sample_time,
                    "valid",
                    step
                )
                for idx in top_indices:
                    valid_scores.append(
                        self._evaluate_formula_with_tensors(
                            formulas_list[idx],
                            feat_tensor,
                            raw_data_cache,
                            target_ret
                        )
                    )
                if valid_scores:
                    valid_scores_tensor = torch.tensor(valid_scores, device=ModelConfig.DEVICE)
                    valid_score = valid_scores_tensor.mean().item()
                    best_idx = int(valid_scores_tensor.argmax().item())
                    if valid_scores_tensor[best_idx].item() > self.best_valid_score:
                        self.best_valid_score = valid_scores_tensor[best_idx].item()
                        self.best_formula = formulas_list[top_indices[best_idx]]
                    last_valid_score = valid_score
            
            # 4. 策略梯度更新
            self._policy_gradient_update(log_probs, entropies, train_scores)
            
            # 5. LoRD更新
            if self.lord_opt:
                self.lord_opt.step()
            
            # 6. 记录日志
            log_valid_score = last_valid_score if valid_score == -999 else valid_score
            self._log_training_step(step, train_scores, log_valid_score, pbar)
            pbar.refresh()
            
            # 7. 保存检查点
            if self.config.save_checkpoints and step % self.config.save_checkpoint_interval == 0 and self.best_formula:
                self._save_checkpoint(step)

            if self.config.eval_cache_clear_interval > 0 and step % self.config.eval_cache_clear_interval == 0:
                self._clear_eval_cache()

            if self.config.gc_interval > 0 and step % self.config.gc_interval == 0:
                gc.collect()
            
            # 8. 早停检查
            if step >= early_stop_start and self.early_stopper.should_stop(self.best_valid_score):
                print(f"\n🛑 早停触发于第 {step} 步")
                break
        
        print(f"\n✅ 训练完成! 最佳验证分数: {self.best_valid_score:.4f}")
        
        # 全周期评估
        self._evaluate_full_period_metrics()
        
        # 最终保存
        self._save_final_results()

        return self.best_formula
    
    def _evaluate_full_period_metrics(self):
        """评估全周期指标 (Train/Valid/Test)"""
        if self.best_formula is None:
            return

        loaders = [
            ("Train", self.train_loader),
            ("Valid", self.valid_loader),
            ("Test", self.test_loader)
        ]
        
        print("-" * 80)
        print(f"{'Period':<10} {'Ann Excess':<15} {'Ann Gross':<15} {'IR':<10} {'Turnover':<15}")
        print("-" * 80)
        
        for name, loader in loaders:
            if loader is None:
                continue
                
            try:
                # Get tensors
                feat_tensor = loader.feat_tensor
                target_ret = loader.target_ret
                
                # Execute VM
                with torch.no_grad():
                    factors = self.vm.execute(self.best_formula, feat_tensor)
                    
                if factors is None:
                    print(f"{name:<10} {'Failed (None)':<15}")
                    continue
                    
                # Calculate metrics using the unified evaluate function
                metrics = self.backtest.evaluate(factors, target_ret)
                
                print(f"{name:<10} {metrics['ann_excess_ret']:<15.2%} {metrics['ann_excess_ret_gross']:<15.2%} {metrics['ir']:<10.4f} {metrics['annual_turnover']:<15.2%}")
                
            except Exception as e:
                print(f"{name:<10} Error: {e}")
        print("-" * 80)
    
    def _generate_formulas_batch(self):
        """生成一批公式"""
        log_probs_list = []
        entropies_list = []
        tokens_list = []
        
        inp = torch.zeros((self.config.batch_size, 1), 
                         dtype=torch.long, device=ModelConfig.DEVICE)
        stack_depth = torch.zeros(self.config.batch_size, 
                                 dtype=torch.long, device=ModelConfig.DEVICE)
        
        for step_idx in range(self.config.max_formula_len):
            logits, _, _ = self.model(inp)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 应用masking
            logits = self._apply_masking(logits, stack_depth, step_idx)
            
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            # 更新堆栈深度
            stack_depth += self.token_change[action]
            
            tokens_list.append(action)
            log_probs_list.append(dist.log_prob(action))
            entropies_list.append(dist.entropy())
            
            inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
        
        formulas = torch.stack(tokens_list, dim=1)
        log_probs = torch.stack(log_probs_list, dim=1)
        entropies = torch.stack(entropies_list, dim=1)
        
        return formulas, log_probs, entropies
    
    def _apply_masking(self, logits, stack_depth, step_idx):
        """应用语法约束masking"""
        batch_size = logits.shape[0]
        arity_req = self.token_arity.unsqueeze(0).expand(batch_size, -1)
        curr_depth = stack_depth.unsqueeze(1)
        
        # 基础约束
        valid_mask = (curr_depth >= arity_req)
        
        # 最终步约束
        if step_idx == self.config.max_formula_len - 1:
            target_change = 1 - curr_depth
            change_val = self.token_change.unsqueeze(0).expand(batch_size, -1)
            final_mask = (change_val == target_change)
            valid_mask = valid_mask & final_mask
        
        # 处理无效情况
        invalid_batch = (valid_mask.sum(dim=1) == 0)
        if invalid_batch.any():
            # 强制使得第一个token (feature 0) 有效
            valid_mask[invalid_batch, 0] = True
            logits[invalid_batch, 0] = 0.0
        
        return logits.masked_fill(~valid_mask, -float('inf'))
    
    def _evaluate_single_vm_task(self, args):
        """单个公式的VM执行任务 (仅执行，不进行同步检查)"""
        formula, feat_tensor = args
        try:
            with torch.no_grad():
                result = self.vm.execute(formula, feat_tensor)
                return result
        except Exception:
            return None

    def _evaluate_formulas_batch(self, formulas, data_loader, sample_assets=None, sample_time=None, step=0):
        scores = torch.full((self.config.batch_size,), self.config.syntax_error_penalty, device=ModelConfig.DEVICE, dtype=torch.float32)
        if isinstance(formulas, torch.Tensor):
            formulas_list = formulas.detach().cpu().tolist()
        else:
            formulas_list = formulas

        feat_tensor, _, target_tensor = self._get_eval_tensors(
            data_loader,
            sample_assets,
            sample_time,
            "train",
            step
        )

        # Serial evaluation loop for simplicity and stability
        for i in range(len(formulas_list)):
            try:
                # This is a simplified serial evaluation logic that combines
                # the logic from _evaluate_formula_with_tensors directly into the batch loop.
                
                # 1. Compute single factor value
                factor_tensor = self.vm.execute(formulas_list[i], feat_tensor)

                if factor_tensor is None:
                    # Syntax error in formula, score is already set to penalty
                    continue

                # 2. Check for validity (const, nan, inf)
                result_std = factor_tensor.std().item()
                if result_std < 1e-6:
                    scores[i] = self.config.constant_penalty
                    continue
                if torch.isnan(factor_tensor).any() or torch.isinf(factor_tensor).any():
                    scores[i] = self.config.constant_penalty
                    continue

                # 3. Calculate IC score
                pred = factor_tensor.reshape(-1)
                target = target_tensor.reshape(-1)

                # Remove NaN values from both pred and target
                valid_mask = ~torch.isnan(pred) & ~torch.isnan(target)
                if valid_mask.sum() < 2:
                    scores[i] = self.config.eval_error_penalty
                    continue
                
                pred = pred[valid_mask]
                target = target[valid_mask]

                # Pearson correlation
                pred_mean = torch.mean(pred)
                target_mean = torch.mean(target)
                vx = pred - pred_mean
                vy = target - target_mean

                numerator = torch.sum(vx * vy)
                denominator = torch.sqrt(torch.sum(vx ** 2) * torch.sum(vy ** 2))
                
                ic_val = 0.0
                if denominator > 1e-8:
                    ic_val = (numerator / denominator).item()

                # 4. Convert IC to reward
                ic_clip = self.config.ic_clip
                if ic_clip is not None:
                    ic_val = max(min(ic_val, ic_clip), -ic_clip)

                if ic_val >= 0:
                    reward = self.config.base_reward + ic_val * self.config.positive_ic_multiplier
                else:
                    reward = self.config.base_reward + ic_val * self.config.negative_ic_multiplier
                
                scores[i] = reward

            except Exception as e:
                if self.config.debug_mode:
                    print(f"Error evaluating formula {i} in batch: {e}")
                # Keep the syntax_error_penalty, which is the default
        
        return scores
    
    def _policy_gradient_update(self, log_probs, entropies, rewards):
        """策略梯度更新"""
        if rewards.std() > 1e-8:
            advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantage = rewards - rewards.mean()
        
        policy_loss = -(log_probs * advantage.unsqueeze(1)).mean()
        entropy_loss = -entropies.mean()
        total_loss = policy_loss + self.config.entropy_coef * entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
    
    def _log_training_step(self, step, train_scores, valid_score, pbar):
        """记录训练步骤"""
        avg_train = train_scores.mean().item()
        std_train = train_scores.std().item()
        
        # 更新进度条
        postfix = {
            'Train': f"{avg_train:.3f}±{std_train:.3f}",
            'BestValid': f"{self.best_valid_score:.3f}"
        }
        
        current_rank = np.nan
        if self.rank_monitor and step % self.config.rank_monitor_interval == 0:
            current_rank = self.rank_monitor.compute()
            postfix['Rank'] = f"{current_rank:.2f}"
            
        # 更新历史
        self.training_history['step'].append(step)
        self.training_history['train_reward'].append(avg_train)
        self.training_history['train_reward_std'].append(std_train)
        self.training_history['valid_score'].append(valid_score)
        self.training_history['stable_rank'].append(current_rank)
        
        pbar.set_postfix(postfix)
        
        # 定期打印详细信息
        if step % self.config.log_interval == 0 and self.config.debug_mode:
            print(f"\n[Step {step}] Train: {avg_train:.4f} ± {std_train:.4f}, "
                  f"Valid: {valid_score:.4f}, Best: {self.best_valid_score:.4f}")
    
    def _save_checkpoint(self, step):
        """保存检查点"""
        checkpoint = {
            'step': step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_valid_score': self.best_valid_score,
            'best_formula': self.best_formula,
            'config': self.config.__dict__
        }

        if self.config.checkpoint_history_max_len > 0:
            history = {}
            for key, values in self.training_history.items():
                history[key] = values[-self.config.checkpoint_history_max_len:]
            checkpoint['training_history'] = history
        
        torch.save(checkpoint, 
                  f"{self.config.output_dir}/checkpoints/step_{step:06d}.pth")
    
    def _save_final_results(self):
        """保存最终结果"""
        if self.config.save_config:
            config_path = f"{self.config.output_dir}/training_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2, default=str)
            print(f"✅ 配置已保存到 {config_path}")
        
        # 2. 保存最佳策略
        if self.best_formula and self.config.save_best_strategy:
            strategy_data = {
                'formula_tokens': self.best_formula,
                'valid_score': self.best_valid_score,
                'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'training_steps': len(self.training_history['step'])
            }
            
            strategy_path = f"{self.config.output_dir}/best_strategy.json"
            with open(strategy_path, 'w') as f:
                json.dump(strategy_data, f, indent=2)
            print(f"✅ 策略已保存到 {strategy_path}")
        
        # 3. 保存模型
        if self.config.save_model_weights:
            model_path = f"{self.config.output_dir}/alphagpt_model.pth"
            torch.save(self.model.state_dict(), model_path)
            print(f"✅ 模型权重已保存到 {model_path}")
        
        # 4. 保存训练历史
        if self.config.save_training_history:
            history_path = f"{self.config.output_dir}/training_history.csv"
            history_df = pd.DataFrame(self.training_history)
            history_df.to_csv(history_path, index=False)
            print(f"✅ 训练历史已保存到 {history_path}")
        
        if self.config.plot_training_history:
            self._plot_training_history()

    def _plot_training_history(self):
        """可视化训练曲线"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置风格
            sns.set_style("whitegrid")
            
            # 创建画布
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. 训练奖励
            df = pd.DataFrame(self.training_history)
            
            axes[0, 0].plot(df['step'], df['train_reward'], label='Train Reward')
            axes[0, 0].fill_between(df['step'], 
                                   df['train_reward'] - df['train_reward_std'],
                                   df['train_reward'] + df['train_reward_std'],
                                   alpha=0.3)
            axes[0, 0].set_title('Training Reward')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Reward')
            
            # 2. 验证分数
            axes[0, 1].plot(df['step'], df['valid_score'], label='Valid Score', color='orange')
            axes[0, 1].set_title('Validation Score')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Score')
            
            # 3. Stable Rank
            if 'stable_rank' in df.columns and not df['stable_rank'].isna().all():
                # 过滤NaN
                valid_rank = df.dropna(subset=['stable_rank'])
                axes[1, 0].plot(valid_rank['step'], valid_rank['stable_rank'], label='Stable Rank', color='green')
                axes[1, 0].set_title('Model Stable Rank')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Rank')
            
            # 保存
            plt.tight_layout()
            plt.savefig(f"{self.config.output_dir}/training_history.png")
            plt.close()
            print(f"✅ 训练曲线已保存到 {self.config.output_dir}/training_history.png")
            
        except Exception as e:
            print(f"⚠️ 无法绘制训练曲线: {e}")

# ============== 辅助类 ==============
class EarlyStopper:
    """早停器"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -float('inf')
    
    def should_stop(self, current_score):
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def test_ic_calculation():
    """测试IC计算是否正常"""
    print("\n🧪 IC计算测试:")
    
    # 创建随机因子 
    n_assets, n_time = 100, 50
    random_factors = torch.randn(n_assets, n_time)
    random_returns = torch.randn(n_assets, n_time) * 0.01
    
    # 计算IC 
    try:
        from .ashare_backtest import AShareBacktest
        bt = AShareBacktest()
        
        # 测试随机数据IC（应该接近0）
        ic, _ = bt.calculate_rank_ic(random_factors, random_returns)
        print(f"  随机因子IC: {ic:.4f} (应该接近0)")
        
        # 测试完美相关数据
        perfect_factors = random_returns * 1.0
        ic, _ = bt.calculate_rank_ic(perfect_factors, random_returns)
        print(f"  完美相关IC: {ic:.4f} (应该接近1)")
        
        # 测试完全负相关
        negative_factors = -random_returns
        ic, _ = bt.calculate_rank_ic(negative_factors, random_returns)
        print(f"  完全负相关IC: {ic:.4f} (应该接近-1)")
    except Exception as e:
        print(f"  ❌ IC测试失败: {e}")
    print("-" * 30)

def run_training(config, save_strategy_path=None, run_state_eval=False):
    print("训练配置:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    print("-" * 50)

    print("\n[加载训练数据]")
    train_loader = QlibDataLoader()
    train_loader.load_data(
        start_time=config.train_start,
        end_time=config.train_end,
        instruments=config.instruments
    )
    print(f"训练集加载完成: {train_loader.feat_tensor.shape}")

    print("\n[加载验证数据]")
    valid_loader = QlibDataLoader()
    valid_loader.load_data(
        start_time=config.valid_start,
        end_time=config.valid_end,
        instruments=config.instruments
    )
    print(f"验证集加载完成: {valid_loader.feat_tensor.shape}")

    print("\n[加载测试数据]")
    test_loader = QlibDataLoader()
    test_loader.load_data(
        start_time=config.test_start,
        end_time=config.test_end,
        instruments=config.instruments
    )
    print(f"测试集加载完成: {test_loader.feat_tensor.shape}")

    print("\n[初始化训练引擎]")
    engine = ConfigurableAShareAlphaEngine(config, train_loader, valid_loader, test_loader)

    print("\n[开始训练]")
    engine.train()

    if engine.best_formula is None:
        print("\n⚠️ 未找到有效公式，使用默认公式 [0]")
        engine.best_formula = [0]

    if save_strategy_path:
        os.makedirs(os.path.dirname(save_strategy_path) or ".", exist_ok=True)
        with open(save_strategy_path, "w") as f:
            json.dump(engine.best_formula, f)
        print(f"✅ 策略已保存到 {save_strategy_path}")

    if run_state_eval:
        try:
            index_df = D.features(["SH000300"], ["$close"], start_time=config.valid_start, end_time=config.valid_end)
            index_close = extract_index_close(index_df)
            instruments = D.instruments(market=config.instruments)
            raw_df = D.features(instruments, ["$close"], start_time=config.valid_start, end_time=config.valid_end)
            factor_scores = engine.vm.execute(engine.best_formula, valid_loader.feat_tensor)
            if factor_scores is not None:
                pred_df = build_pred_df_from_factor_scores(factor_scores, raw_df)
                evaluate_state_metrics(pred_df, raw_df, index_close, output_dir="outputs", prefix="valid_state", window=60)
                print("✅ Valid state evaluation saved to outputs")
        except Exception as e:
            print(f"⚠️ Valid state evaluation skipped: {e}")

    return engine, train_loader, valid_loader

# ============== 主函数 ==============
if __name__ == "__main__":
    test_ic_calculation()
    config = TrainingConfig(
        batch_size=128,
        train_steps=1000,
        max_formula_len=20,
        learning_rate=5e-4,
        entropy_coef=0.01,
        train_start="2021-01-01",
        train_end="2023-12-31",
        valid_start="2024-01-01",
        valid_end="2024-12-31",
        instruments="csi300",
        syntax_error_penalty=-1.0,
        constant_penalty=-0.2,
        eval_error_penalty=-0.5,
        base_reward=0.0,
        positive_ic_multiplier=5.0,
        negative_ic_multiplier=0.3,
        ic_clip=0.5,
        debug_mode=True,
        validation_interval=50,
        validation_top_k=1,
        train_sample_assets=128,
        train_sample_time=256,
        valid_sample_assets=256,
        valid_sample_time=256,
        log_interval=10,
        save_checkpoint_interval=50,
        early_stop_patience=30,
    )

    engine, train_loader, valid_loader = run_training(
        config,
        save_strategy_path=f"{config.output_dir}/best_strategy.json",
        run_state_eval=True,
    )

    if engine.best_formula:
        print(f"\n最佳公式: {engine.best_formula}")
        print(f"📊 验证分数: {engine.best_valid_score:.4f}")
        print("\n[在训练集和验证集上评估最佳公式]")
        train_score = engine.evaluate_formula(engine.best_formula, train_loader)
        valid_score = engine.evaluate_formula(engine.best_formula, valid_loader)
        print(f"训练集得分: {train_score:.4f}")
        print(f"验证集得分: {valid_score:.4f}")
        print(f"过拟合程度: {(train_score - valid_score):.4f}")
