from typing import Union, Sequence, Tuple
import torch
import einops
import itertools


def make_tuple_nd(x: Union[int, Sequence[int]], ndim: int) -> Sequence[int]:
    """
    Takes hetrogeneous input of sequences of list and build consistent
    tuples.

    ## Exapmles
    >>> make_tuple_nd(x=7, ndim=3)
    (7, 7, 7)
    >>> make_tuple_nd(x=[2, 5], ndim=2)
    (2, 5)
    >>> make_tuple_nd(x=1, ndim=1)
    (1,)
    >>> make_tuple_nd(x=(1, 2), ndim=3) # Should raise ValueError
    Traceback (most recent call last):
        ...
    ValueError: Input sequence length 2 != ndim 3
    >>> make_tuple_nd(x="hello", ndim=2) # Should raise TypeError
    Traceback (most recent call last):
        ...
    TypeError: Input must be int or sequence, got <class 'str'>
    """

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
    """
    Transforms a nd-shape into patches of `kernel_size` and moves the
    newly added dimensions to the batch dimension. Returns the batched
    tensor and the new spatial dimensions after batching.

    ## Exapmles
    >>> import torch
    >>> # Example for ndim=2 (Image)
    >>> tensor_2d = torch.arange(1 * 3 * 4 * 4).reshape(1, 3, 4, 4) # B, C, H, W
    >>> kernel_2d = (2, 2)
    >>> batched_2d, dims_2d = batch_nd(tensor_2d, ndim=2, kernel_size=kernel_2d)
    >>> batched_2d.shape # B, (H/kH * W/kW), (kH * kW), C
    torch.Size([1, 4, 4, 3])
    >>> dims_2d # [H/kH, W/kW]
    [2, 2]
    >>> # Example for ndim=3 (Volume)
    >>> tensor_3d = torch.arange(1 * 2 * 4 * 4 * 4).reshape(1, 2, 4, 4, 4) # B, C, D, H, W
    >>> kernel_3d = 2
    >>> batched_3d, dims_3d = batch_nd(tensor_3d, ndim=3, kernel_size=kernel_3d)
    >>> batched_3d.shape # B, (D/kD * H/kH * W/kW), (kD * kH * kW), C
    torch.Size([1, 8, 8, 2])
    >>> dims_3d # [D/kD, H/kH, W/kW]
    [2, 2, 2]
    """

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
    """
    Reverses the `batch_nd` operation, transforming patches back into
    spatial dimensions.

    ## Examples
    >>> import torch
    >>> # Continuing 2D example from batch_nd
    >>> tensor_2d = torch.arange(1 * 3 * 4 * 4).reshape(1, 3, 4, 4) # B, C, H, W
    >>> kernel_2d = (2, 2)
    >>> batched_2d, dims_2d = batch_nd(tensor_2d, ndim=2, kernel_size=kernel_2d)
    >>> batched_2d.shape
    torch.Size([1, 4, 4, 3])
    >>> dims_2d
    [2, 2]
    >>> unbatched_2d = unbatch_nd(batched_2d, ndim=2, kernel_size=kernel_2d, dimensions=dims_2d)
    >>> unbatched_2d.shape # Should match original tensor_2d shape
    torch.Size([1, 3, 4, 4])
    >>> torch.equal(unbatched_2d, tensor_2d)
    True
    """

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
    """
    Performs a circular shift on the spatial dimensions of a tensor.

    ## Examples
    >>> import torch
    >>> # Simple 2D example
    >>> tensor = torch.arange(1 * 1 * 3 * 3).reshape(1, 1, 3, 3)
    >>> tensor
    tensor([[[[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8]]]])
    >>> shifted = shift_nd(tensor, ndim=2, shift_size=(1, 1)) # Shift H by -1, W by -1
    >>> shifted # Rows/cols shifted up/left
    tensor([[[[4, 5, 3],
              [7, 8, 6],
              [1, 2, 0]]]])
    """
    if ndim <= 0:
        raise ValueError("ndim must be positive.")

    shift_size = make_tuple_nd(shift_size, ndim)
    shift_size = [-s for s in shift_size]
    return torch.roll(x, shifts=shift_size, dims=list(range(2, 2 + ndim)))


def unshift_nd(
    x: torch.Tensor, ndim: int, shift_size: Union[int, Sequence[int]]
) -> torch.Tensor:
    """
    Reverses the circular shift performed by `shift_nd`.

    ## Examples
    >>> import torch
    >>> # Continuing 2D example from shift_nd
    >>> tensor = torch.arange(1 * 1 * 3 * 3).reshape(1, 1, 3, 3)
    >>> shifted = shift_nd(tensor, ndim=2, shift_size=(1, 1))
    >>> shifted
    tensor([[[[4, 5, 3],
              [7, 8, 6],
              [1, 2, 0]]]])
    >>> unshifted = unshift_nd(shifted, ndim=2, shift_size=(1, 1)) # Shift H by +1, W by +1
    >>> unshifted # Should match original tensor
    tensor([[[[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8]]]])
    >>> torch.equal(unshifted, tensor)
    True
    """
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
) -> torch.Tensor:
    """
    Generates a batched shift mask for an ND-shape. Non-overlapping regions
    after a shift should not receive attention, hence a mask.
    """

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
    """
    Generates absolute coordinates for an ND-shape. The output is a grid
    with the shape (D d0 d1 d2 ... dN).
    """

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


def generate_relative_coordinate_indices_nd(
    ndim: int,
    dimensions: Union[int, Sequence[int]],
    max_distances: Union[int, Sequence[int]] = None,
    normalise: bool = True,
    device=None,
) -> torch.Tensor:
    """
    The function generates based on the relative positions indices. These
    indices can be used to access (prod(max_dist) + 1) embeddings.
    """

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
