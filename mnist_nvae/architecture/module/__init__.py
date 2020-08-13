from mnist_nvae.architecture.module.swish import Swish
from mnist_nvae.architecture.module.center_slice_residual import CenterSliceResidual
from mnist_nvae.architecture.module.residual_stack import ResidualStack
from mnist_nvae.architecture.module.random_fourier import (
    RandomFourier, FourierSampleDecoder
)
from mnist_nvae.architecture.module.variational import (
    Variational,
    RelativeVariational,
    VariationalBlock,
    RelativeVariationalBlock,
)
from mnist_nvae.architecture.module.squeeze_excitation import SqueezeExcitation
