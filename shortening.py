import torch

# Reduced-precision types whose reductions must accumulate in float32.
_LOW_PRECISION = (torch.bfloat16, torch.float16)


def accum_dtype(dtype):
    """Dtype to accumulate pooling reductions in.

    bfloat16 carries 8 mantissa bits, so both halves of a pooled mean round at
    ~0.4%: a `1/n` weight is off by up to 0.2% (a *systematic* bias on every
    pooled value) and a running sum re-rounds at every term. torch already
    accumulates bfloat16 `sum`/`einsum` in float32 internally; doing the same
    for the normalisation and for the cache's incremental means keeps pooling
    accurate and, because both paths then reduce identically, keeps them
    numerically equivalent. Storage stays in the model dtype -- only the
    arithmetic is widened. A no-op for float32 and float64.
    """
    return torch.float32 if dtype in _LOW_PRECISION else dtype


def check_closes(c1, c2, c3):
    """Validate the close-event contract shared by the naive and cached paths.

    Every close event must be binary and cumulative (`c3 <= c2 <= c1`): a
    level-2 event also closes level 1, a level-3 event also closes levels 1 and
    2. `Tokenizer.group` enforces this, but the model is also callable with
    arrays built by hand or loaded from disk, and a violation makes the naive
    and cached paths disagree silently (they build different pooled tensors),
    so it is rejected here rather than propagated.
    """
    for name, c in (("c1", c1), ("c2", c2), ("c3", c3)):
        if not bool(((c == 0) | (c == 1)).all()):
            raise ValueError(f"{name} must be binary (0/1) close events")
    if not bool((c2 <= c1).all()) or not bool((c3 <= c2).all()):
        raise ValueError("close events must be cumulative: c3 <= c2 <= c1")


def level_boundaries(c1, c2, c3, dtype=torch.float, validate=True):
    """Derive the pooling-boundary array at each hierarchy level.

    Inputs are causal close-events at token resolution (cumulative: c3<=c2<=c1):
        c1, c2, c3: B x T  (1 where the position closes that level)

    Returns (in `dtype`, which must match the hidden states they pool):
        bnd1: B x T          boundaries used to pool tokens -> level 1
        bnd2: B x (K1max+1)  boundaries used to pool level 1 -> level 2
        bnd3: B x (K2max+1)  boundaries used to pool level 2 -> level 3

    bnd2/bnd3 live at the *pooled* resolution and include the leading null-group
    slot (index 0, always 0). bnd2[j] marks whether the j-th completed level-1
    group also closed a level-2 group; bnd3[m] the analogous fact for level 2.
    This is obtained by scattering the coarser close-event onto the slot each
    boundary token occupies in the pooled tensor (cumsum of the finer event).
    """
    c1 = c1.long()
    c2 = c2.long()
    c3 = c3.long()
    if validate:
        check_closes(c1, c2, c3)
    B = c1.size(0)

    bnd1 = c1.to(dtype)

    k1_max = int(c1.sum(dim=1).max().item())
    slot1 = torch.cumsum(c1, dim=1) * c1  # boundary token -> its slot 1..K1; else 0
    bnd2 = torch.zeros(B, k1_max + 1, device=c1.device, dtype=dtype)
    bnd2.scatter_(1, slot1, (c2 * c1).to(dtype))
    bnd2[:, 0] = 0

    k2_max = int(c2.sum(dim=1).max().item())
    slot2 = torch.cumsum(c2, dim=1) * c2
    bnd3 = torch.zeros(B, k2_max + 1, device=c1.device, dtype=dtype)
    bnd3.scatter_(1, slot2, (c3 * c2).to(dtype))
    bnd3[:, 0] = 0

    return bnd1, bnd2, bnd3


# `common` and `final` build the dense membership matrix and are used only by
# the `*_dense` reference implementations at the bottom of this file.
def final(foo,
          upsample):
    """
        Input:
            B x L x S
    """
    autoregressive = foo != 0
    lel = 1 - foo

    lel[autoregressive] = 0

    dim = 2 if upsample else 1

    # Members per slot. An unused (padded) slot has none, and a zero numerator,
    # so clamping the divisor to 1 leaves it zero -- while dividing by the exact
    # count, rather than count + eps, makes pooling an exact mean. The eps was
    # invisible in float32 but is the dominant naive-vs-cached difference in
    # float64, where the cached path's running mean has no such factor.
    # Normalise in the accumulation dtype: see `accum_dtype`.
    lel = lel.to(accum_dtype(lel.dtype))
    lel = lel / lel.sum(dim=dim, keepdim=True).clamp(min=1)

    return lel


def common(boundaries, upsample=False):
    boundaries = boundaries.clone()

    # int(): `.item()` on float boundaries yields a Python float, which would
    # make the arange below float32 and silently upcast the whole computation
    # (breaking bfloat16/float16 hidden states).
    n_segments = int(boundaries.sum(dim=-1).max().item())

    if upsample:
        n_segments += 1

    if n_segments == 0:
        return None

    tmp = torch.zeros_like(
        boundaries
    ).unsqueeze(2) + torch.arange(
        start=0,
        end=n_segments,
        device=boundaries.device
    )

    # Count boundaries in int64: a bfloat16 cumsum of 1s stops being exact past
    # 256, which would silently corrupt the group indices on long sequences.
    hh1 = boundaries.long().cumsum(1)

    if not upsample:
        hh1 = hh1 - boundaries.long()

    foo = tmp - hh1.unsqueeze(-1)

    return foo


def downsample_dense(boundaries, hidden, null_group):
    """
        Downsampling (reference implementation)

        - The first element of boundaries tensor is always 0 and doesn't matter
        - 1 closes the current group after this position
        - We append an extra "null" group at the beginning
        - We discard an incomplete trailing group

        Input:
            boundaries: B x L
            hidden: L x B x D
        Output:
            shortened_hidden: S x B x D

    Materialises the full B x L x S membership matrix and contracts it against
    the hidden states. Kept as the reference `downsample` is checked against;
    see `downsample` for why that matrix is not needed.
    """

    foo = common(boundaries, upsample=False)  # B x L x S

    if foo is None:
        return null_group.repeat(1, hidden.size(1), 1)
    else:
        bar = final(foo=foo, upsample=False)  # B x L x S

        # Reduce in `bar`'s (accumulation) dtype, store in the hidden dtype.
        shortened_hidden = torch.einsum(
            'lbd,bls->sbd', hidden.to(bar.dtype), bar).to(hidden.dtype)
        shortened_hidden = torch.cat(
            [null_group.repeat(1, hidden.size(1), 1), shortened_hidden], dim=0
        )

        return shortened_hidden


def upsample_dense(boundaries, shortened_hidden):
    """
        Upsampling (reference implementation)

        - The first element of boundaries tensor is always 0 and doesn't matter
        - 1 closes the current group after this position
        - The newly completed group becomes visible at its closing position

        Input:
            boundaries: B x L
            shortened_hidden: S x B x D
        Output:
            upsampled_hidden: L x B x D

    Materialises the full B x L x S matrix -- which for upsampling is one-hot in
    S -- and contracts it. Kept as the reference `upsample` is checked against.
    """

    foo = common(boundaries, upsample=True)  # B x L x S
    bar = final(foo, upsample=True)  # B x L x S

    return torch.einsum(
        'sbd,bls->lbd', shortened_hidden.to(bar.dtype), bar
    ).to(shortened_hidden.dtype)


def downsample(boundaries, hidden, null_group):
    """
        Downsampling

        - The first element of boundaries tensor is always 0 and doesn't matter
        - 1 closes the current group after this position
        - We append an extra "null" group at the beginning
        - We discard an incomplete trailing group

        Input:
            boundaries: B x L
            hidden: L x B x D
        Output:
            shortened_hidden: S x B x D

    Groups are ragged -- different lengths, and a different count per batch
    member -- so this is a segment mean, not a mean over any axis. Each position
    is scattered into its group's slot and the totals are divided by the counts,
    which needs O(L) index memory instead of the reference's B x L x S
    membership matrix (512 MiB at L=8192, B=8). Summing first and dividing once
    also means there is no 1/n weight to round, so the group mean is exact in
    reduced precision. `downsample_dense` is the equivalent reference; the tests
    check the two against each other.
    """
    B = hidden.size(1)
    n_segments = int(boundaries.sum(dim=-1).max().item())

    if n_segments == 0:
        return null_group.repeat(1, B, 1)

    acc = accum_dtype(hidden.dtype)
    # Group index of each position: boundaries closed strictly before it. The
    # highest index is a member's trailing incomplete group, dropped below.
    group = (boundaries.long().cumsum(1) - boundaries.long()).transpose(0, 1)
    index = group.unsqueeze(-1).expand(-1, -1, hidden.size(2))

    total = torch.zeros(n_segments + 1, B, hidden.size(2),
                        device=hidden.device, dtype=acc)
    total.scatter_add_(0, index, hidden.to(acc))
    count = torch.zeros(n_segments + 1, B, device=hidden.device, dtype=acc)
    count.scatter_add_(0, group, torch.ones_like(group, dtype=acc))

    # A slot with no members (a member with fewer groups than the batch maximum)
    # keeps its zero numerator, matching the reference.
    mean = (total / count.clamp(min=1).unsqueeze(-1)).to(hidden.dtype)
    return torch.cat([null_group.repeat(1, B, 1), mean[:n_segments]], dim=0)


def upsample(boundaries, shortened_hidden):
    """
        Upsampling

        - The first element of boundaries tensor is always 0 and doesn't matter
        - 1 closes the current group after this position
        - The newly completed group becomes visible at its closing position

        Input:
            boundaries: B x L
            shortened_hidden: S x B x D
        Output:
            upsampled_hidden: L x B x D

    Each position reads exactly one group -- the most recently completed one --
    so this is a gather, not a weighted sum, and it is exact in any dtype.
    `upsample_dense` is the equivalent reference.
    """
    index = boundaries.long().cumsum(1).transpose(0, 1)          # L x B
    index = index.unsqueeze(-1).expand(-1, -1, shortened_hidden.size(2))
    return shortened_hidden.gather(0, index)
