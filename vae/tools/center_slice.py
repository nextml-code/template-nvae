import torch


def center_slice_cat(tensors, dim):
    for slice_dim in range(tensors[0].ndim):
        if slice_dim != dim:
            tensors = center_slice_dim(tensors, slice_dim)
    return torch.cat(tensors, dim=dim)


def center_slice_dim(tensors, dim):
    tensor_lengths = [tensor.shape[dim] for tensor in tensors]
    slice_length = min(tensor_lengths)
    slice_starts = [(length - slice_length) // 2 for length in tensor_lengths]
    # print('tensor_lengths:', tensor_lengths)
    return [
        tensor.narrow(dim, slice_start, slice_length)
        for slice_start, tensor in zip(slice_starts, tensors)
    ]


# TODO: refactor
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
        center_slice_dim2(larger, smaller)
        for larger, smaller in zip(larger_shape, smaller_shape)
    ]


def center_slice_dim2(larger, smaller):
    start = (larger - smaller) // 2
    return slice(start, start + smaller)
