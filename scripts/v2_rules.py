"""Persistable causal rule fixtures; names include their semantic version."""
from tokenizer import register_group_rule


@register_group_rule('v2-test-previous-x7-v1')
def previous_x7(token_id, default, state):
    if default == (1, 0, 0) and state['previous_token_id'] != 9:
        return 0, 0, 0
    if token_id == 11:  # x9 creates a leaf close.
        return 1, 0, 0
    return default
