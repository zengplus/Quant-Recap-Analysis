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
        valid_loaders=None,
        valid_agg="mean",
        valid_score_mode="legacy",
        use_lord_regularization=True,
        lord_decay_rate=1e-3,
        lord_num_iterations=5,
        token_mode="postfix",
    ):
        self.train_loader = train_loader
        if valid_loaders is None:
            valid_loaders = []
        if valid_loader is not None:
            valid_loaders = [valid_loader] + list(valid_loaders)
        self.valid_loaders = [v for v in valid_loaders if v is not None]
        self.valid_agg = str(valid_agg).lower().strip() if valid_agg is not None else "mean"
        self.valid_score_mode = str(valid_score_mode).lower().strip() if valid_score_mode is not None else "legacy"
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
        self.bt = AShareBacktest(
            rebalance_freq="M",
            topk=getattr(ModelConfig, "TOPK", 10),
            signal_lag=getattr(ModelConfig, "SIGNAL_LAG", 1),
            turnover_penalty=getattr(ModelConfig, "TURNOVER_PENALTY", 0.0),
            abs_weight=getattr(ModelConfig, "ABS_WEIGHT", 1.0),
            alpha_weight_bull=getattr(ModelConfig, "ALPHA_WEIGHT_BULL", 1.0),
            alpha_weight_bear=getattr(ModelConfig, "ALPHA_WEIGHT_BEAR", 0.3),
            market_regime_threshold=getattr(ModelConfig, "MARKET_REGIME_THRESHOLD", 0.0),
            neg_excess_penalty=getattr(ModelConfig, "NEG_EXCESS_PENALTY", 0.0),
            neg_port_penalty=getattr(ModelConfig, "NEG_PORT_PENALTY", 0.0),
            min_excess_weight=getattr(ModelConfig, "MIN_EXCESS_WEIGHT", 0.0),
            beta_target=getattr(ModelConfig, "BETA_TARGET", 1.0),
            beta_min_bull=getattr(ModelConfig, "BETA_MIN_BULL", 0.9),
            beta_penalty_weight_bull=getattr(ModelConfig, "BETA_PENALTY_WEIGHT_BULL", 0.0),
            beta_min_penalty_weight_bull=getattr(ModelConfig, "BETA_MIN_PENALTY_WEIGHT_BULL", 0.0),
            beta_penalty_weight_bull_obj=getattr(ModelConfig, "BETA_PENALTY_WEIGHT_BULL_OBJ", 0.0),
            beta_min_penalty_weight_bull_obj=getattr(ModelConfig, "BETA_MIN_PENALTY_WEIGHT_BULL_OBJ", 0.0),
            bull_excess_min=getattr(ModelConfig, "BULL_EXCESS_MIN", 0.0),
            bull_total_min=getattr(ModelConfig, "BULL_TOTAL_MIN", 0.0),
            bear_excess_min=getattr(ModelConfig, "BEAR_EXCESS_MIN", 0.0),
            bear_total_min=getattr(ModelConfig, "BEAR_TOTAL_MIN", -0.3),
            bull_excess_target=getattr(ModelConfig, "BULL_EXCESS_TARGET", 0.0),
            bull_excess_hinge_weight_score=getattr(ModelConfig, "BULL_EXCESS_HINGE_WEIGHT_SCORE", 0.0),
            bull_excess_hinge_weight_obj=getattr(ModelConfig, "BULL_EXCESS_HINGE_WEIGHT_OBJ", 0.0),
        )
        self.bt.tail_seg_fraction = float(getattr(ModelConfig, "TAIL_SEG_FRACTION", 0.2))
        self.best_valid_score = -float('inf')
        self.best_formula = None
        self.training_history = {
            'step': [],
            'train_reward': [],
            'valid_score': [],
            'stable_rank': []
        }
        self.token_logit_bias = None
        bias_map = getattr(ModelConfig, "TOKEN_LOGIT_BIAS", None)
        if isinstance(bias_map, dict) and bias_map:
            bias = torch.zeros(self.model.vocab_size, device=ModelConfig.DEVICE, dtype=torch.float32)
            for k, v in bias_map.items():
                try:
                    idx = self.model.vocab.index(str(k))
                except Exception:
                    continue
                try:
                    bias[idx] = bias[idx] + float(v)
                except Exception:
                    continue
            self.token_logit_bias = bias

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

    def evaluate_formula_objective(self, formula, loader, allow_repair=True):
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
        score, _, details = self.bt.evaluate(
            res,
            loader.raw_data_cache,
            loader.target_ret,
            dates=loader.dates,
            return_details=True,
        )
        if details is not None and isinstance(details, dict) and "objective_score" in details:
            try:
                return float(details.get("objective_score"))
            except Exception:
                return float(score.item())
        return float(score.item())

    def evaluate_formula_objective_details(self, formula, loader, allow_repair=True):
        if self.token_mode == "prefix":
            res = self.vm.execute_prefix(formula, loader.feat_tensor)
        else:
            res = self.vm.execute(formula, loader.feat_tensor)
        if res is None:
            if not allow_repair:
                return -5.0, None
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
                return -5.0, None
        if torch.isnan(res).all():
            return -5.0, None
        score, _, details = self.bt.evaluate(
            res,
            loader.raw_data_cache,
            loader.target_ret,
            dates=loader.dates,
            return_details=True,
        )
        if details is not None and isinstance(details, dict) and "objective_score" in details:
            try:
                return float(details.get("objective_score")), details
            except Exception:
                return float(score.item()), details
        return float(score.item()), details

    def evaluate_formula_valid(self, formula, allow_repair=True):
        if not self.valid_loaders:
            return -999.0
        if self.valid_score_mode == "bullbear_rule":
            scores = []
            bull_scores = []
            bear_scores = []
            for v in self.valid_loaders:
                s, d = self.evaluate_formula_objective_details(formula, v, allow_repair=allow_repair)
                scores.append(float(s))
                is_bull = False
                bull_excess_ok = True
                objective_pass = True
                if isinstance(d, dict):
                    try:
                        is_bull = bool(d.get("bench_is_bull"))
                    except Exception:
                        is_bull = False
                    if is_bull:
                        try:
                            bull_excess_ok = bool(float(d.get("excess_total_return_pct")) > float(getattr(ModelConfig, "VALID_BULL_EXCESS_MIN_PCT", 0.0)))
                        except Exception:
                            bull_excess_ok = True
                    try:
                        objective_pass = bool(d.get("objective_pass"))
                    except Exception:
                        objective_pass = True
                if is_bull and bool(getattr(ModelConfig, "VALID_ENFORCE_BULL_EXCESS", False)) and (not bull_excess_ok):
                    return -9999.0
                if is_bull and bool(getattr(ModelConfig, "VALID_ENFORCE_BULL_OBJECTIVE_PASS", False)) and (not objective_pass):
                    return -9999.0
                if (not is_bull) and bool(getattr(ModelConfig, "VALID_ENFORCE_BEAR_OBJECTIVE_PASS", False)) and (not objective_pass):
                    return -9999.0
                if is_bull:
                    bull_scores.append(float(s))
                else:
                    bear_scores.append(float(s))
        else:
            scores = [self.evaluate_formula(formula, v, allow_repair=allow_repair) for v in self.valid_loaders]
        if not scores:
            return -999.0
        if self.valid_agg in ("bull_min_bear_mean", "bullmin_bearmean", "bullminbearmean"):
            if self.valid_score_mode != "bullbear_rule":
                return float(sum(scores) / float(len(scores)))
            if not bull_scores and not bear_scores:
                return float(sum(scores) / float(len(scores)))
            bw = float(getattr(ModelConfig, "VALID_BULL_WEIGHT", 1.0))
            rw = float(getattr(ModelConfig, "VALID_BEAR_WEIGHT", 0.3))
            if not (bw == bw):
                bw = 1.0
            if not (rw == rw):
                rw = 0.3
            if bw < 0.0:
                bw = 0.0
            if rw < 0.0:
                rw = 0.0
            bull_term = float(min(bull_scores)) if bull_scores else float(sum(scores) / float(len(scores)))
            bear_term = float(sum(bear_scores) / float(len(bear_scores))) if bear_scores else 0.0
            return float(bw * bull_term + rw * bear_term)
        if self.valid_agg == "min":
            return float(min(scores))
        if self.valid_agg == "mix":
            w = float(getattr(ModelConfig, "VALID_MIX_MIN_WEIGHT", 0.7))
            w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)
            mean_s = float(sum(scores) / float(len(scores)))
            min_s = float(min(scores))
            return float(w * min_s + (1.0 - w) * mean_s)
        if self.valid_agg == "median":
            s = sorted(float(x) for x in scores)
            m = len(s) // 2
            return float(s[m]) if (len(s) % 2 == 1) else float((s[m - 1] + s[m]) / 2.0)
        return float(sum(scores) / float(len(scores)))

    def train(self):
        print("🚀 Starting Alpha Mining...")
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        use_objective = bool(getattr(ModelConfig, "TRAIN_USE_OBJECTIVE", False))
        require_train_obj_pass = bool(getattr(ModelConfig, "VALID_REQUIRE_TRAIN_OBJECTIVE_PASS", False))
        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            log_probs = []
            tokens_list = []
            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                if torch.isnan(logits).any():
                    logits = torch.nan_to_num(logits, nan=0.0)
                if self.token_logit_bias is not None:
                    logits = logits + self.token_logit_bias.unsqueeze(0)
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
                if use_objective:
                    score = float(self.evaluate_formula_objective(formula, self.train_loader, allow_repair=True))
                else:
                    score = float(self.evaluate_formula(formula, self.train_loader, allow_repair=True))
                rewards[i] = score
                if score > batch_best_train_score:
                    batch_best_train_score = score
                    batch_best_idx = i
            current_valid_score = -999
            if self.valid_loaders:
                rerank_topn = int(getattr(ModelConfig, "VALID_RERANK_TOPN", 1))
                if rerank_topn < 1:
                    rerank_topn = 1
                rerank_topn = int(min(rerank_topn, bs))
                cand_idx = torch.topk(rewards, k=rerank_topn, largest=True).indices.tolist()
                best_i = -1
                best_s = -float("inf")
                reward_w = float(getattr(ModelConfig, "VALID_REWARD_WEIGHT", 0.0))
                reward_scale = float(getattr(ModelConfig, "VALID_REWARD_SCALE", 100.0))
                if not (reward_w == reward_w):
                    reward_w = 0.0
                if reward_w < 0.0:
                    reward_w = 0.0
                if reward_w > 1.0:
                    reward_w = 1.0
                if not (reward_scale == reward_scale) or reward_scale <= 0.0:
                    reward_scale = 100.0
                for i in cand_idx:
                    formula = seqs[int(i)].tolist()
                    if require_train_obj_pass:
                        _, d_train = self.evaluate_formula_objective_details(formula, self.train_loader, allow_repair=True)
                        train_ok = True
                        if isinstance(d_train, dict):
                            try:
                                train_ok = bool(d_train.get("objective_pass"))
                            except Exception:
                                train_ok = True
                        if not train_ok:
                            continue
                    s = float(self.evaluate_formula_valid(formula, allow_repair=True))
                    if reward_w > 0.0:
                        if s > -1e8 and s < 1e8:
                            rewards[int(i)] = rewards[int(i)] + reward_w * (float(s) / float(reward_scale))
                    if s > best_s:
                        best_s = s
                        best_i = int(i)
                current_valid_score = float(best_s) if best_i >= 0 else -999.0
                if best_i >= 0 and current_valid_score > self.best_valid_score:
                    self.best_valid_score = float(current_valid_score)
                    self.best_formula = seqs[int(best_i)].tolist()
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
