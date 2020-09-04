import torch
import torch.nn.functional as F
from math import log


@torch.jit.script
def log_cosh(x):
    return x + F.softplus(-2 * x) - log(2)
