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
            default_val = feat_tensor[:, 0, :]
            for token in formula_tokens:
                token = int(token)
                if token < self.feat_offset:
                    stack.append(feat_tensor[:, token, :])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity:
                        missing = arity - len(stack)
                        for _ in range(missing):
                            stack.append(default_val)
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()
                    func = self.op_map[token]
                    res = func(*args)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
                    stack.append(res)
                else:
                    # print(f"Unknown token {token}")
                    return None
            if len(stack) == 1:
                return stack[0]
            else:
                # print(f"Invalid final stack size: {len(stack)}")
                return None
        except Exception:
            return None

    def execute_prefix(self, formula_tokens, feat_tensor):
        stack = []
        try:
            default_val = feat_tensor[:, 0, :]
            for token in reversed(formula_tokens):
                token = int(token)
                if token < self.feat_offset:
                    stack.append(feat_tensor[:, token, :])
                    continue
                if token not in self.op_map:
                    continue
                arity = self.arity_map[token]
                if len(stack) < arity:
                    missing = arity - len(stack)
                    for _ in range(missing):
                        stack.append(default_val)
                args = []
                for _ in range(arity):
                    args.append(stack.pop())
                func = self.op_map[token]
                res = func(*args)
                if torch.isnan(res).any() or torch.isinf(res).any():
                    res = torch.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
                stack.append(res)
            return stack[-1] if stack else None
        except Exception:
            return None

    def repair_postfix(self, formula_tokens):
        stack_size = 0
        valid_tokens = []
        # Assume ADD is the first operator (index self.feat_offset)
        # This is a heuristic.
        ADD_TOKEN = self.feat_offset 
        
        for t in formula_tokens:
            t = int(t)
            if t < self.feat_offset:
                stack_size += 1
                valid_tokens.append(t)
            elif t in self.arity_map:
                arity = self.arity_map[t]
                if stack_size >= arity:
                    stack_size -= (arity - 1)
                    valid_tokens.append(t)
        
        # Ensure stack size is exactly 1
        if stack_size == 0:
            valid_tokens.append(0) # Default feature 0
            stack_size = 1
            
        while stack_size > 1:
            valid_tokens.append(ADD_TOKEN)
            stack_size -= 1
            
        return valid_tokens

    def repair_prefix(self, formula_tokens):
        # Prefix is equivalent to Postfix when processed from right to left
        rev_tokens = list(reversed(formula_tokens))
        repaired_rev = self.repair_postfix(rev_tokens)
        return list(reversed(repaired_rev))
