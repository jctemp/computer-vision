from typing import Union, Sequence, Tuple
import torch
import einops
import itertools


def make_tuple_nd(x: Union[int, Sequence[int]], ndim: int) -> Sequence[int]:
    """Helper function to create a tuple of size ndim."""
    if isinstance(x, int):
        return tuple([x] * ndim)
    elif isinstance(x, (list, tuple)):
        if len(x) == ndim:
            return tuple(x)
        else:
            raise ValueError(f"Input sequence length {len(x)} != ndim {ndim}")
    else:
        raise TypeError(f"Input must be int or sequence, got {type(x)}")


def batch_nd(
    x: torch.Tensor,
    ndim: int,
    kernel_size: Union[int, Sequence[int]],
) -> Tuple[torch.Tensor, Sequence[int]]:
    if ndim <= 0:
        raise ValueError("ndim must be positive.")

    kernel_size = make_tuple_nd(kernel_size, ndim)
    tensor_shape = x.size()

    spatial_dims_in = " ".join([f"(d{i} k{i})" for i in range(ndim)])
    spatial_dims_out = " ".join([f"d{i}" for i in range(ndim)])
    kernel_channels_out = " ".join([f"k{i}" for i in range(ndim)])

    rearrange_in_pattern = (
        f"b c {spatial_dims_in} -> b ({spatial_dims_out}) ({kernel_channels_out}) c"
    )
    rearrange_params = {f"k{i}": ks for i, ks in enumerate(kernel_size)}

    x = einops.rearrange(x, rearrange_in_pattern, **rearrange_params)
    dimensions = [d // k for d, k in zip(tensor_shape[2:], kernel_size)]

    return (x, dimensions)


def unbatch_nd(
    x: torch.Tensor,
    ndim: int,
    kernel_size: Union[int, Sequence[int]],
    dimensions: Union[int, Sequence[int]],
) -> Tuple[torch.Tensor, Sequence[int]]:
    if ndim <= 0:
        raise ValueError("ndim must be positive.")

    kernel_size = make_tuple_nd(kernel_size, ndim)
    dimensions = make_tuple_nd(dimensions, ndim)

    spatial_dims_in = " ".join([f"d{i}" for i in range(ndim)])
    kernel_channels_in = " ".join([f"k{i}" for i in range(ndim)])
    spatial_dims_out = " ".join([f"(d{i} k{i})" for i in range(ndim)])

    rearrange_in_pattern = (
        f"b ({spatial_dims_in}) ({kernel_channels_in}) c -> b c {spatial_dims_out}"
    )
    rearrange_kernel_params = {f"k{i}": ks for i, ks in enumerate(kernel_size)}
    rearrange_dims_params = {f"d{i}": ds for i, ds in enumerate(dimensions)}

    x = einops.rearrange(
        x, rearrange_in_pattern, **rearrange_kernel_params, **rearrange_dims_params
    )

    return x


def shift_nd(
    x: torch.Tensor, ndim: int, shift_size: Union[int, Sequence[int]]
) -> torch.Tensor:
    if ndim <= 0:
        raise ValueError("ndim must be positive.")

    shift_size = make_tuple_nd(shift_size, ndim)
    shift_size = [-s for s in shift_size]
    return torch.roll(x, shifts=shift_size, dims=list(range(2, 2 + ndim)))


def unshift_nd(
    x: torch.Tensor, ndim: int, shift_size: Union[int, Sequence[int]]
) -> torch.Tensor:
    if ndim <= 0:
        raise ValueError("ndim must be positive.")

    shift_size = make_tuple_nd(shift_size, ndim)
    return torch.roll(x, shifts=shift_size, dims=list(range(2, 2 + ndim)))


def generate_shift_nd_mask(
    ndim: int,
    kernel_size: Union[int, Sequence[int]],
    dimensions: Union[int, Sequence[int]],
    shift_size: Union[int, Sequence[int]],
    device=None,
):
    if ndim <= 0:
        raise ValueError("ndim must be positive.")

    kernel_size = make_tuple_nd(kernel_size, ndim)
    dimensions = make_tuple_nd(dimensions, ndim)
    shift_size = make_tuple_nd(shift_size, ndim)

    # 1. Generate index map
    index_map = torch.zeros(dimensions, device=device)

    slices = [
        [
            (0, -k),
            (-k, -s),
            (-s, None),
        ]
        for k, s in zip(kernel_size, shift_size)
    ]

    # 1.1. Build indices
    count = 0
    for multi_dim_slice in itertools.product(*slices):
        multi_dim_slice = tuple(slice(start, stop) for start, stop in multi_dim_slice)

        # Assign regional id
        index_map[multi_dim_slice] = count
        count += 1

    index_map_batched: torch.Tensor
    (index_map_batched, _) = batch_nd(
        index_map.unsqueeze(0).unsqueeze(0), ndim, kernel_size
    )
    index_map_batched = index_map_batched.squeeze(-1).squeeze(0)

    # 2. Compare indices to find window differences
    ids = index_map_batched.unsqueeze(1) - index_map_batched.unsqueeze(2)

    # 3. Mask non-equal window sections
    ids = ids.masked_fill(ids != 0, True).masked_fill(ids == 0, False)

    # 4. Retrieve broadcastable shape for attention computation
    ids = einops.rearrange(ids, "bw ... -> () bw () ...")

    return ids.to(torch.bool)


def generate_absolute_coordinates_nd(
    ndim: int,
    dimensions: Union[int, Sequence[int]],
    device=None,
):
    if ndim <= 0:
        raise ValueError("ndim must be positive.")

    dimensions = make_tuple_nd(dimensions, ndim)

    spatial_dim = [f"d{i}" for i in range(ndim)]
    spatial_dims = " ".join(spatial_dim)
    spatial_params = {p: v for p, v in zip(spatial_dim, dimensions)}

    indices = [
        einops.repeat(
            torch.arange(val, device=device),
            f"{dim} -> {spatial_dims}",
            **spatial_params,
        )
        for dim, val in zip(spatial_dim, dimensions)
    ]

    return torch.stack(indices, dim=0)


def generate_relative_attention_coordinates_nd(
    ndim: int,
    dimensions: Union[int, Sequence[int]],
    max_distances: Union[int, Sequence[int]] = None,
    normalise: bool = True,
    device=None,
):
    if ndim <= 0:
        raise ValueError("ndim must be positive.")

    dimensions = make_tuple_nd(dimensions, ndim)
    max_distances = (
        make_tuple_nd(max_distances, ndim) if max_distances is not None else dimensions
    )
    offsets = [0] + list(itertools.accumulate((2 * d + 1 for d in max_distances[:-1])))

    spatial_dim = [f"d{i}" for i in range(ndim)]
    spatial_dims = " ".join(spatial_dim)
    spatial_params = {p: v for p, v in zip(spatial_dim, dimensions)}

    indices = [
        einops.repeat(
            torch.arange(val, dtype=torch.float, device=device),
            f"{dim} -> {spatial_dims}",
            **spatial_params,
        ).flatten()
        for dim, val in zip(spatial_dim, dimensions)
    ]

    relative_indices = [index.unsqueeze(0) - index.unsqueeze(1) for index in indices]

    relative_indices_shifted = [
        torch.clamp(index, -max_dist, max_dist) + max_dist + offset
        for index, max_dist, offset in zip(relative_indices, max_distances, offsets)
    ]

    if normalise:
        relative_indices_shifted = [
            index / (max_dist - 1 if max_dist > 1 else max_dist)
            for index, max_dist in zip(relative_indices, max_distances)
        ]

    torch.stack(relative_indices_shifted, dim=0).shape
