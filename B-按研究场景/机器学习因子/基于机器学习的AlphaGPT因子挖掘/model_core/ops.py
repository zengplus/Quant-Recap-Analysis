""" 
 File: ops.py
 Date: 2026-01-17
 Description: 算子定义库。定义了虚拟机支持的所有算子（如加减乘除、时序延迟、逻辑门等）及其对应的 PyTorch 实现，部分算子经过 JIT 优化。
 From: https://github.com/imbue-bit/AlphaGPT
 """ 
import torch

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

@torch.jit.script
def _op_jump(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)

@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

@torch.jit.script
def _op_tsum20(x: torch.Tensor) -> torch.Tensor:
    w = 20
    if x.size(1) <= 1:
        return x
    pad = torch.zeros((x.size(0), w - 1), device=x.device, dtype=x.dtype)
    x_pad = torch.cat([pad, x], dim=1)
    return x_pad.unfold(1, w, 1).sum(dim=-1)

OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('GATE', _op_gate, 3),
    ('JUMP', _op_jump, 1),
    ('DECAY', _op_decay, 1),
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('MAX3', lambda x: torch.max(x, torch.max(_ts_delay(x,1), _ts_delay(x,2))), 1),
    ('TSUM20', _op_tsum20, 1),
]
