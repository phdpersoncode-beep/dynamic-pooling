import torch


def level_boundaries(c1, c2, c3):
    """Derive the pooling-boundary array at each hierarchy level.

    Inputs are causal close-events at token resolution (cumulative: c3<=c2<=c1):
        c1, c2, c3: B x T  (1 where the position closes that level)

    Returns:
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
    B = c1.size(0)

    bnd1 = c1.float()

    k1_max = int(c1.sum(dim=1).max().item())
    slot1 = torch.cumsum(c1, dim=1) * c1  # boundary token -> its slot 1..K1; else 0
    bnd2 = torch.zeros(B, k1_max + 1, device=c1.device)
    bnd2.scatter_(1, slot1, (c2 * c1).float())
    bnd2[:, 0] = 0.0

    k2_max = int(c2.sum(dim=1).max().item())
    slot2 = torch.cumsum(c2, dim=1) * c2
    bnd3 = torch.zeros(B, k2_max + 1, device=c1.device)
    bnd3.scatter_(1, slot2, (c3 * c2).float())
    bnd3[:, 0] = 0.0

    return bnd1, bnd2, bnd3


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

    lel = lel / (lel.sum(dim=dim, keepdim=True) + 1e-9)

    return lel


def common(boundaries, upsample=False):
    boundaries = boundaries.clone()

    n_segments = boundaries.sum(dim=-1).max().item()

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

    hh1 = boundaries.cumsum(1)

    if not upsample:
        hh1 -= boundaries

    foo = tmp - hh1.unsqueeze(-1)

    return foo


def downsample(boundaries, hidden, null_group):
    """
        Downsampling

        - The first element of boundaries tensor is always 0 and doesn't matter
        - 1 starts a new group
        - We append an extra "null" group at the beginning
        - We discard last group because it won't be used (in terms of upsampling)

        Input:
            boundaries: B x L
            hidden: L x B x D
        Output:
            shortened_hidden: S x B x D
    """

    foo = common(boundaries, upsample=False)  # B x L x S

    if foo is None:
        return null_group.repeat(1, hidden.size(1), 1)
    else:
        bar = final(foo=foo, upsample=False)  # B x L x S

        shortened_hidden = torch.einsum('lbd,bls->sbd', hidden, bar)
        shortened_hidden = torch.cat(
            [null_group.repeat(1, hidden.size(1), 1), shortened_hidden], dim=0
        )

        return shortened_hidden


def upsample(boundaries, shortened_hidden):
    """
        Upsampling

        - The first element of boundaries tensor is always 0 and doesn't matter
        - 1 starts a new group
        - i-th group can be upsampled only to the tokens from (i+1)-th group, otherwise there's a leak

        Input:
            boundaries: B x L
            shortened_hidden: S x B x D
        Output:
            upsampled_hidden: L x B x D
    """

    foo = common(boundaries, upsample=True)  # B x L x S
    bar = final(foo, upsample=True)  # B x L x S

    return torch.einsum('sbd,bls->lbd', shortened_hidden, bar)
