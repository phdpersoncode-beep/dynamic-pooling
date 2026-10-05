import pytest
import torch
from numerics import compare_logits


def test_margin_certificate_distinguishes_ties_and_actual_mismatches():
    reference = torch.tensor([[3., 1.], [1., 1.], [1.01, 1.]])
    cached = torch.tensor([[2.9, 1.1], [1., 1.], [1., 1.02]])
    report = compare_logits(reference, cached)
    assert report['greedy_mismatches'] == 1
    assert report['rows_with_margin_over_twice_error'] == 1
    assert report['rows_without_margin_certificate'] == 2
    assert not report['within_historical_threshold']


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_values_cannot_pass_audit(value):
    with pytest.raises(ValueError, match='nonfinite'):
        compare_logits(torch.tensor([1., value]), torch.tensor([1., 2.]))


def test_difference_is_measured_without_low_precision_subtraction():
    a = torch.tensor([1., -256.], dtype=torch.bfloat16)
    b = torch.tensor([1., 1.], dtype=torch.bfloat16)
    assert compare_logits(a, b)['max_abs_error'] == 257.
