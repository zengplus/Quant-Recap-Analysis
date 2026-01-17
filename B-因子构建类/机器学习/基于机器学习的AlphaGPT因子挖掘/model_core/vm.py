""" 
 File: vm.py
 Date: 2026-01-17
 Description: 堆栈虚拟机核心。定义了 StackVM 类，负责解析和执行基于堆栈的因子公式，支持多种算子操作。
 From: https://github.com/imbue-bit/AlphaGPT
 """ 
import torch
from .ops import OPS_CONFIG
from .factors import FeatureEngineer

class StackVM:
    def __init__(self):
        self.feat_offset = FeatureEngineer.INPUT_DIM
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

    def execute(self, formula_tokens, feat_tensor):
        stack = []
        try:
            for token in formula_tokens:
                token = int(token)
                if token < self.feat_offset:
                    stack.append(feat_tensor[:, token, :])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity: 
                        # print(f"Stack underflow for token {token} (Arity {arity}, Stack {len(stack)})")
                        return None
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()
                    func = self.op_map[token]
                    res = func(*args)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                else:
                    # print(f"Unknown token {token}")
                    return None
            if len(stack) == 1:
                return stack[0]
            else:
                # print(f"Invalid final stack size: {len(stack)}")
                return None
        except Exception as e:
            # print(f"VM Execution Error: {e}")
            return None