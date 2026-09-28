import torch

from shortening import downsample, level_boundaries, upsample


def test_level_boundaries_move_coarse_events_to_pooled_slots():
    c1 = torch.tensor([[0, 1, 0, 1, 1]])
    c2 = torch.tensor([[0, 0, 0, 1, 0]])
    c3 = torch.tensor([[0, 0, 0, 1, 0]])

    bnd1, bnd2, bnd3 = level_boundaries(c1, c2, c3)

    assert bnd1.tolist() == [[0, 1, 0, 1, 1]]
    assert bnd2.tolist() == [[0, 0, 1, 0]]
    assert bnd3.tolist() == [[0, 1]]


def test_incomplete_group_is_not_downsampled():
    boundaries = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.float)
    hidden = torch.tensor([1, 3, 5, 7, 9], dtype=torch.float).view(5, 1, 1)
    null = torch.tensor([[[-1.0]]])

    shortened = downsample(boundaries, hidden, null)

    assert shortened[:, 0, 0].tolist() == [-1.0, 2.0]


def test_completed_group_becomes_visible_at_its_closing_position():
    boundaries = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.float)
    shortened = torch.tensor([-1.0, 2.0]).view(2, 1, 1)

    expanded = upsample(boundaries, shortened)

    assert expanded[:, 0, 0].tolist() == [-1.0, 2.0, 2.0, 2.0, 2.0]
