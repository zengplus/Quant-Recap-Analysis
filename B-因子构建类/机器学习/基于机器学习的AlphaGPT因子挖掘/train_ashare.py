""" 
 File: train_ashare.py
 Date: 2026-01-17
 Description: A股 Alpha 挖掘训练入口。包含 AShareAlphaEngine 类，负责初始化 AlphaGPT 模型、加载数据、执行训练循环、生成公式并在 A 股回测环境中评估因子表现。
 """ 
import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json
import os
from model_core.config import ModelConfig
from model_core.alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from model_core.vm import StackVM
from model_core.qlib_loader import QlibDataLoader
from model_core.ashare_backtest import AShareBacktest

class AShareAlphaEngine:
    """
    训练引擎 (Train/Valid Split Version)
    """
    def __init__(self, train_loader, valid_loader=None, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        # 1. 数据加载器
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        # 2. 初始化模型 (AlphaGPT Transformer)
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        
        # 标准优化器
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # 3. LoRD (Low-Rank Decay) 正则化器
        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None
        
        # 4. 虚拟机与回测计算器
        self.vm = StackVM()
        self.bt = AShareBacktest()
        
        # 5. 状态记录
        self.best_valid_score = -float('inf')
        self.best_formula = None
        self.training_history = {
            'step': [],
            'train_reward': [],
            'valid_score': [],
            'stable_rank': []
        }

    def evaluate_formula(self, formula, loader):
        """Helper to evaluate a single formula on a specific loader"""
        res = self.vm.execute(formula, loader.feat_tensor)
        if res is None: return -5.0 # Syntax Error
        if res.std() < 1e-4: return -2.0 # Constant
        score, _ = self.bt.evaluate(res, loader.raw_data_cache, loader.target_ret)
        return score.item()

    def train(self):
        print("🚀 Starting Alpha Mining...")
        
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        
        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            
            log_probs = []
            tokens_list = []
            
            # --- 1. 生成公式 (Generate Formulas) ---
            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                if torch.isnan(logits).any():
                    logits = torch.nan_to_num(logits, nan=0.0)
                
                dist = Categorical(logits=logits)
                action = dist.sample()
                
                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
            
            seqs = torch.stack(tokens_list, dim=1)
            
            # --- 2. 训练集评估 (Evaluate on Train Set) ---
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            
            # Keep track of best formula in this batch to validate
            batch_best_idx = -1
            batch_best_train_score = -float('inf')
            
            for i in range(bs):
                formula = seqs[i].tolist()
                score = self.evaluate_formula(formula, self.train_loader)
                rewards[i] = score
                
                # Only consider valid formulas (score > -2.0)
                if score > -2.0 and score > batch_best_train_score:
                    batch_best_train_score = score
                    batch_best_idx = i
            
            # --- 3. 验证集评估 (Validation) ---
            # 仅验证本 Batch 中表现最好的公式，以节省计算资源
            current_valid_score = -999
            if self.valid_loader and batch_best_idx >= 0:
                best_formula_in_batch = seqs[batch_best_idx].tolist()
                current_valid_score = self.evaluate_formula(best_formula_in_batch, self.valid_loader)
                
                # Update Global Best based on VALIDATION score
                if current_valid_score > -2.0 and current_valid_score > self.best_valid_score:
                    self.best_valid_score = current_valid_score
                    self.best_formula = best_formula_in_batch
            
            # --- 4. 策略梯度更新 (Policy Gradient Update) ---
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            loss = 0
            for t in range(len(log_probs)):
                loss += -log_probs[t] * adv
            loss = loss.mean()
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            if self.use_lord:
                self.lord_opt.step()
            
            # --- 5. Logging ---
            avg_reward = rewards.mean().item()
            postfix_dict = {'TrainRew': f"{avg_reward:.3f}", 'ValidBest': f"{self.best_valid_score:.3f}"}
            
            if self.use_lord and step % 10 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
            
            self.training_history['step'].append(step)
            self.training_history['train_reward'].append(avg_reward)
            self.training_history['valid_score'].append(current_valid_score)
            
            pbar.set_postfix(postfix_dict)
        
        print(f"\n✅ Training Completed. Best Valid Score: {self.best_valid_score:.4f}")

if __name__ == "__main__":
    # 优化配置以加快运行速度
    print("⚡️ Optimizing config for fast run...")
    ModelConfig.BATCH_SIZE = 512
    ModelConfig.TRAIN_STEPS = 100

    # 定义时间段
    TRAIN_START = '2021-01-01'
    TRAIN_END   = '2022-12-31'
    VALID_START = '2023-01-01'
    VALID_END   = '2023-06-30'
    
    print(f"📅 Data Split:\n  Train: {TRAIN_START} ~ {TRAIN_END}\n  Valid: {VALID_START} ~ {VALID_END}")

    # 1. 训练集加载器
    train_loader = QlibDataLoader()
    try:
        print("\n[Loading Train Data]")
        train_loader.load_data(start_time=TRAIN_START, end_time=TRAIN_END, instruments='csi300')
    except Exception as e:
        print(f"❌ Train data loading failed: {e}")
        exit(1)

    # 2. 验证集加载器
    valid_loader = QlibDataLoader()
    try:
        print("\n[Loading Valid Data]")
        valid_loader.load_data(start_time=VALID_START, end_time=VALID_END, instruments='csi300')
    except Exception as e:
        print(f"❌ Valid data loading failed: {e}")
        exit(1)

    # 3. 运行训练
    engine = AShareAlphaEngine(train_loader, valid_loader, use_lord_regularization=True)
    engine.train()
    
    # 4. Fallback Logic
    if engine.best_formula is None:
        print("⚠️ Warning: Training did not find a valid formula.")
        print("⚠️ Using fallback formula [0] (Feature 0).")
        engine.best_formula = [0]

    # 5. 保存结果
    os.makedirs("outputs", exist_ok=True)
    
    # 保存策略
    with open("outputs/best_ashare_strategy.json", "w") as f:
        json.dump(engine.best_formula, f)
    print(f"✅ Strategy saved to outputs/best_ashare_strategy.json")
    
    # 保存模型权重
    torch.save(engine.model.state_dict(), "outputs/alphagpt_model.pth")
    print("✅ Model weights saved to outputs/alphagpt_model.pth")
