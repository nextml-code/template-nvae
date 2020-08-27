

def center_slice_like(x, like):
    return x.__getitem__(
        center_slice(x.shape, like.shape)
    )


def center_slice(larger_shape, smaller_shape):
    if len(larger_shape) != len(smaller_shape):
        raise ValueError(
            'Expected shapes to be same length but got shapes: '
            f'{larger_shape}, {smaller_shape}'
        )
    return [
        center_slice_dim(larger, smaller)
        for larger, smaller in zip(larger_shape, smaller_shape)
    ]


def center_slice_dim(larger, smaller):
    start = (larger - smaller) // 2
    return slice(start, start + smaller)
