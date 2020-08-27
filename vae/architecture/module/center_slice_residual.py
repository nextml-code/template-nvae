from torch import nn

from vae import tools


class CenterSliceResidual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        head = self.module(x)
        return head + tools.center_slice_like(x, head)
