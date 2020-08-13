import torch
import torch.nn.functional as F


@torch.jit.script
def mish(x):
    return x * torch.tanh(F.softplus(x))
