from typing import List, Optional, Tuple
import itertools

import torch
import einops

Input2d = int | List[int] | Tuple[int, int]
Input3d = int | List[int] | Tuple[int, int, int]
Input4d = int | List[int] | Tuple[int, int, int, int]
InputNd = Input2d | Input3d | Input3d
TupleNd = Tuple[int, int] | Tuple[int, int, int] | Tuple[int, int, int, int]


def make_tuple_2d(obj: Input2d):
    if isinstance(obj, int):
        obj = (obj, obj)
    elif isinstance(obj, list):
        if len(obj) != 2:
            raise ValueError(f"obj is a list of length {len(obj)}: expected 2")
        else:
            obj = tuple(obj)
    return obj


def make_tuple_3d(obj: Input3d):
    if isinstance(obj, int):
        obj = (obj, obj, obj)
    elif isinstance(obj, list):
        if len(obj) != 3:
            raise ValueError(f"obj is a list of length {len(obj)}: expected 3")
        else:
            obj = tuple(obj)
    return obj


def make_tuple_4d(obj: Input4d):
    if isinstance(obj, int):
        obj = (obj, obj, obj, obj)
    elif isinstance(obj, list):
        if len(obj) != 4:
            raise ValueError(f"obj is a list of length {len(obj)}: expected 4")
        else:
            obj = tuple(obj)
    return obj


def compute_absolute_coordinates_2d(size: Input2d, device=None) -> torch.Tensor:
    size = make_tuple_2d(size)

    indices_height = einops.repeat(
        torch.arange(size[0], device=device),
        "c -> () c w",
        w=size[1],
    )
    indices_width = einops.repeat(
        torch.arange(size[1], device=device),
        "c -> () h c",
        h=size[0],
    )
    coords = torch.concat([indices_height, indices_width], dim=0)

    return coords


def compute_absolute_coordinates_3d(size: Input3d, device=None) -> torch.Tensor:
    size = make_tuple_3d(size)

    indices_depth = einops.repeat(
        torch.arange(size[0], device=device),
        "c -> () c h w",
        h=size[1],
        w=size[2],
    )
    indices_height = einops.repeat(
        torch.arange(size[1], device=device),
        "c -> () d c w",
        d=size[0],
        w=size[2],
    )
    indices_width = einops.repeat(
        torch.arange(size[2], device=device),
        "c -> () d h c",
        d=size[0],
        h=size[1],
    )
    coords = torch.concat([indices_depth, indices_height, indices_width], dim=0)

    return coords


def compute_absolute_coordinates_4d(size: Input4d, device=None) -> torch.Tensor:
    size = make_tuple_4d(size)

    indices_n = einops.repeat(
        torch.arange(size[0], device=device),
        "c -> () c d h w",
        d=size[1],
        h=size[2],
        w=size[3],
    )
    indices_depth = einops.repeat(
        torch.arange(size[1], device=device),
        "c -> () n c h w",
        n=size[0],
        h=size[2],
        w=size[3],
    )
    indices_height = einops.repeat(
        torch.arange(size[2], device=device),
        "c -> () n d c w",
        n=size[0],
        d=size[1],
        w=size[3],
    )
    indices_width = einops.repeat(
        torch.arange(size[3], device=device),
        "c -> () n d h c",
        n=size[0],
        d=size[1],
        h=size[2],
    )
    coords = torch.concat(
        [indices_n, indices_depth, indices_height, indices_width], dim=0
    )

    return coords


def compute_relative_positions_2d(
    size: Input2d,
    max_distance: Optional[Input2d] = None,
    normalise: bool = True,
    device=None,
) -> torch.Tensor:
    size = make_tuple_2d(size)
    max_distance = make_tuple_2d(max_distance) if max_distance is not None else size
    offsets = [0] + list(itertools.accumulate((2 * d + 1 for d in max_distance[:-1])))

    h, w = size
    mh, mw = max_distance
    oh, ow = offsets

    idx_h = einops.repeat(
        torch.arange(h, dtype=torch.float, device=device), "c -> c w", w=w
    ).flatten()
    idx_w = einops.repeat(
        torch.arange(w, dtype=torch.float, device=device), "c -> h c", h=h
    ).flatten()

    r_idx_h = idx_h.unsqueeze(0) - idx_h.unsqueeze(1)
    r_idx_w = idx_w.unsqueeze(0) - idx_w.unsqueeze(1)

    r_idx_h = torch.clamp(r_idx_h, -mh, mh) + mh + oh
    r_idx_w = torch.clamp(r_idx_w, -mw, mw) + mw + ow

    if normalise:
        r_idx_h /= mh - 1 if mh > 1 else mh
        r_idx_w /= mw - 1 if mw > 1 else mw

    return torch.stack([r_idx_h, r_idx_w], dim=-1)


def compute_relative_positions_3d(
    size: Input3d,
    max_distance: Optional[Input3d] = None,
    normalise: bool = True,
    device=None,
) -> torch.Tensor:
    size = make_tuple_3d(size)
    max_distance = make_tuple_3d(max_distance) if max_distance is not None else size
    offsets = [0] + list(itertools.accumulate((2 * d + 1 for d in max_distance[:-1])))

    d, h, w = size
    md, mh, mw = max_distance
    od, oh, ow = offsets

    idx_d = einops.repeat(
        torch.arange(d, dtype=torch.float, device=device), "c -> c h w", h=h, w=w
    ).flatten()
    idx_h = einops.repeat(
        torch.arange(h, dtype=torch.float, device=device), "c -> d c w", d=d, w=w
    ).flatten()
    idx_w = einops.repeat(
        torch.arange(w, dtype=torch.float, device=device), "c -> d h c", d=d, h=h
    ).flatten()

    r_idx_d = idx_d.unsqueeze(0) - idx_d.unsqueeze(1)
    r_idx_h = idx_h.unsqueeze(0) - idx_h.unsqueeze(1)
    r_idx_w = idx_w.unsqueeze(0) - idx_w.unsqueeze(1)

    r_idx_d = torch.clamp(r_idx_d, -md, md) + md + od
    r_idx_h = torch.clamp(r_idx_h, -mh, mh) + mh + oh
    r_idx_w = torch.clamp(r_idx_w, -mw, mw) + mw + ow

    if normalise:
        r_idx_d /= md - 1 if md > 1 else md
        r_idx_h /= mh - 1 if mh > 1 else mh
        r_idx_w /= mw - 1 if mw > 1 else mw

    return torch.stack([r_idx_d, r_idx_h, r_idx_w], dim=-1)


def compute_relative_positions_4d(
    size: Input4d,
    max_distance: Optional[Input4d] = None,
    normalise: bool = True,
    device=None,
) -> torch.Tensor:
    size = make_tuple_4d(size)
    max_distance = make_tuple_4d(max_distance) if max_distance is not None else size
    offsets = [0] + list(itertools.accumulate((2 * d + 1 for d in max_distance[:-1])))

    n, d, h, w = size
    mn, md, mh, mw = max_distance
    on, od, oh, ow = offsets

    idx_n = einops.repeat(
        torch.arange(d, dtype=torch.float, device=device), "c -> c d h w", d=d, h=h, w=w
    ).flatten()
    idx_d = einops.repeat(
        torch.arange(d, dtype=torch.float, device=device), "c -> n c h w", n=n, h=h, w=w
    ).flatten()
    idx_h = einops.repeat(
        torch.arange(h, dtype=torch.float, device=device), "c -> n d c w", n=n, d=d, w=w
    ).flatten()
    idx_w = einops.repeat(
        torch.arange(w, dtype=torch.float, device=device), "c -> n d h c", n=n, d=d, h=h
    ).flatten()

    r_idx_n = idx_n.unsqueeze(0) - idx_n.unsqueeze(1)
    r_idx_d = idx_d.unsqueeze(0) - idx_d.unsqueeze(1)
    r_idx_h = idx_h.unsqueeze(0) - idx_h.unsqueeze(1)
    r_idx_w = idx_w.unsqueeze(0) - idx_w.unsqueeze(1)

    r_idx_n = torch.clamp(r_idx_n, -mn, mn) + mn + on
    r_idx_d = torch.clamp(r_idx_d, -md, md) + md + od
    r_idx_h = torch.clamp(r_idx_h, -mh, mh) + mh + oh
    r_idx_w = torch.clamp(r_idx_w, -mw, mw) + mw + ow

    if normalise:
        r_idx_n /= mn - 1 if mn > 1 else mn
        r_idx_d /= md - 1 if md > 1 else md
        r_idx_h /= mh - 1 if mh > 1 else mh
        r_idx_w /= mw - 1 if mw > 1 else mw

    return torch.stack([r_idx_n, r_idx_d, r_idx_h, r_idx_w], dim=-1)
