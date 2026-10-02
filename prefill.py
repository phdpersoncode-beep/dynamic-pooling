"""Build a continuation cache from one unchanged full-prefix forward pass."""
import torch
from hourglass import STACK_NAMES


@torch.no_grad()
def prefill_batched(model, data, c1, c2, c3, max_len=None):
    """Return (last logits, cache) for a nonempty time x batch prompt.

    Each member's completed groups occupy contiguous cache slots. Later steps
    may add padded slots; attention already counts real ordinal positions.
    This uses temporary forward hooks: do not call concurrently on one model.
    Stateful tokenizer continuation remains the caller's responsibility.
    """
    if model.training:
        raise ValueError('prefill requires eval mode')
    if data.ndim != 2 or 0 in data.shape:
        raise ValueError('expected nonempty time x batch prompt')
    if data.device != next(model.parameters()).device:
        raise ValueError('prompt and model must be on the same device')
    if max_len is not None and max_len < len(data):
        raise ValueError('max_len must cover the prompt')
    captured, trace, handles = {}, {}, []
    for name, layers in model.stacks.items():
        for index, layer in enumerate(layers):
            def capture(module, inputs, output, key=(name, index)):
                _, k, v = output.chunk(3, dim=-1)
                captured[key] = [x.reshape(*x.shape[:2], model.n_head, model.d_head) for x in (k, v)]
            handles.append(layer.dec_attn.qkv_net.register_forward_hook(capture))
    try:
        logits = model(data, c1, c2, c3, _trace=trace)[-1:].clone()
    finally:
        for handle in handles: handle.remove()

    batch = data.shape[1]
    state = model.init_state_batched(batch, max_len=max_len or len(data), device=data.device)
    n0 = torch.full((batch,), len(data), device=data.device, dtype=torch.long)
    n1, n2, n3 = [c.long().sum(0) + 1 for c in (c1, c2, c3)]
    lengths = dict(zip(STACK_NAMES, [n0, n1, n2, n3, n2, n1, n0]))
    model._ensure_pos(state, max(int(n.max()) for n in lengths.values()))
    for name, layers in model.stacks.items():
        if not layers: continue
        fill = int(lengths[name].max())
        model._ensure_cap(state, name, fill)
        state['valid'][name].zero_()
        state['valid'][name][:fill] = torch.arange(fill, device=data.device)[:, None] < lengths[name]
        state['fill'][name] = fill
        for index in range(len(layers)):
            for store, value in zip(('k', 'v'), captured[name, index]):
                state[store][name][index][:fill].copy_(value[:fill])

    for level, name, length, boundary in zip((1, 2, 3), ('h0', 'h1', 'h2'),
                                            (n0, n1, n2), trace['boundaries']):
        hidden = trace[name].to(state['accum_dtype'])
        positions = torch.arange(len(hidden), device=data.device)[:, None]
        last_close = torch.where(boundary.T.bool(), positions, -1).amax(0)
        members = (positions > last_close) & (positions < length)
        state[f'l{level}_sum'] = (hidden * members[..., None]).sum(0, keepdim=True)
        state[f'l{level}_cnt'] = members.sum(0)
    members = torch.arange(batch, device=data.device)
    for name, length in [('h3', n3), ('e2', n2), ('f1', n1)]:
        state[name + '_last'] = trace[name][length - 1, members].unsqueeze(0).clone()
    return logits, state
