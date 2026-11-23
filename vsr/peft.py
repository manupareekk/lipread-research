"""Parameter-efficient speaker adaptation for Auto-AVSR.

Modes:
  lora      LoRA on encoder and decoder attention projections
  lora_enc  LoRA on encoder attention only (carries almost all of the gain)
  lora_dec  LoRA on decoder attention only
  bitfit    biases and norm parameters only
  frontend  3D-CNN front end and its projection only
  full      every parameter (BatchNorm/dropout in train mode)
  full_eval every parameter, BatchNorm stats and dropout frozen
"""
import math, re, torch, torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.in_features, self.out_features = base.in_features, base.out_features
        self.A = nn.Parameter(torch.empty(r, base.in_features)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t().to(x.dtype) @ self.B.t().to(x.dtype)) * self.scale


ATT = r"\.(self_attn|src_attn)\.linear_(q|k|v|out)$"


def lora_targets(mode):
    pre = {"lora": ("encoder.encoders.", "decoder.decoders."), "lora_enc": ("encoder.encoders.",),
           "lora_dec": ("decoder.decoders.",)}[mode]
    return lambda n: n.startswith(pre) and re.search(ATT, n)


def apply(model, mode, r=8):
    """Configure trainable params for an adaptation mode; returns list of trainable param names."""
    if mode in ("full", "full_eval"):
        for p in model.parameters(): p.requires_grad = True
        return [n for n, _ in model.named_parameters()]
    for p in model.parameters(): p.requires_grad = False
    if mode.startswith("lora"):
        sel = lora_targets(mode)
        for n, m in list(model.named_modules()):
            if isinstance(m, nn.Linear) and sel(n):
                parent, attr = model.get_submodule(n.rsplit(".", 1)[0]), n.rsplit(".", 1)[1]
                setattr(parent, attr, LoRALinear(m, r))
        keep = lambda n: n.endswith(".A") or n.endswith(".B")
    elif mode == "bitfit":
        keep = lambda n: n.endswith("bias") or "norm" in n
    elif mode == "frontend":
        keep = lambda n: n.startswith(("frontend.", "proj_encoder."))
    else:
        raise ValueError(mode)
    names = []
    for n, p in model.named_parameters():
        if keep(n): p.requires_grad = True; names.append(n)
    return names


def freeze_stats(model):
    """Keep BatchNorm running stats and dropout fixed while training."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout)): m.eval()
