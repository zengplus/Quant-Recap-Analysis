import torch
from torch.distributions import Categorical
from tqdm import tqdm
from .config import ModelConfig
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .ashare_backtest import AShareBacktest


class AShareAlphaEngine:
    def __init__(
        self,
        train_loader,
        valid_loader=None,
        use_lord_regularization=True,
        lord_decay_rate=1e-3,
        lord_num_iterations=5,
        token_mode="postfix",
    ):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.token_mode = token_mode
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
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
        self.vm = StackVM()
        self.bt = AShareBacktest(rebalance_freq="M")
        self.best_valid_score = -float('inf')
        self.best_formula = None
        self.training_history = {
            'step': [],
            'train_reward': [],
            'valid_score': [],
            'stable_rank': []
        }

    def evaluate_formula(self, formula, loader, allow_repair=True):
        if self.token_mode == "prefix":
            res = self.vm.execute_prefix(formula, loader.feat_tensor)
        else:
            res = self.vm.execute(formula, loader.feat_tensor)
        if res is None:
            if not allow_repair:
                return -5.0
            if self.token_mode == "prefix":
                repaired = self.vm.repair_prefix(formula)
                res = self.vm.execute_prefix(repaired, loader.feat_tensor)
            else:
                repaired = self.vm.repair_postfix(formula)
                res = self.vm.execute(repaired, loader.feat_tensor)
            if res is None:
                fallback = [0]
                if self.token_mode == "prefix":
                    res = self.vm.execute_prefix(fallback, loader.feat_tensor)
                else:
                    res = self.vm.execute(fallback, loader.feat_tensor)
            if res is None:
                return -5.0
        if torch.isnan(res).all():
            return -5.0
        score, _ = self.bt.evaluate(res, loader.raw_data_cache, loader.target_ret, dates=loader.dates)
        return score.item()

    def train(self):
        print("🚀 Starting Alpha Mining...")
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            log_probs = []
            tokens_list = []
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
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            batch_best_idx = -1
            batch_best_train_score = -float('inf')
            for i in range(bs):
                formula = seqs[i].tolist()
                score = self.evaluate_formula(formula, self.train_loader, allow_repair=False)
                rewards[i] = score
                if score > batch_best_train_score:
                    batch_best_train_score = score
                    batch_best_idx = i
            current_valid_score = -999
            if self.valid_loader and batch_best_idx >= 0:
                best_formula_in_batch = seqs[batch_best_idx].tolist()
                current_valid_score = self.evaluate_formula(best_formula_in_batch, self.valid_loader, allow_repair=True)
                if current_valid_score > self.best_valid_score:
                    self.best_valid_score = current_valid_score
                    self.best_formula = best_formula_in_batch
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
