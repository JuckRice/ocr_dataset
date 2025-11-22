import math

class CosineWithWarmup:
    """Cosine LR with linear warmup. min_lr_ratio controls the final LR floor vs base LR."""
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.01):
        self.opt = optimizer
        self.warmup = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in self.opt.param_groups]

    def step(self):
        self.step_num += 1
        for i, g in enumerate(self.opt.param_groups):
            base = self.base_lrs[i]
            if self.step_num <= self.warmup:
                lr = base * self.step_num / max(1, self.warmup)
            else:
                progress = (self.step_num - self.warmup) / max(1, self.max_steps - self.warmup)
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                lr = base * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine)
            g["lr"] = lr
