"""List-based semantic oracle: no production grouping/pooling/cache helpers."""
import torch


def partition(events):
    groups, pending, visible = [], [], []
    for index, closes in enumerate(events):
        pending.append(index)
        if closes:
            groups.append(pending)
            pending = []
        visible.append(len(groups))
    return groups, pending, visible


def layout(symbols):
    events = [[s in allowed for s in symbols] for allowed in
              (("b1", "b2", "b3"), ("b2", "b3"), ("b3",))]
    result = []
    current = events[0]
    for level in range(3):
        groups, pending, visible = partition(current)
        result.append((groups, pending, visible))
        if level == 0:
            current = [False] + [events[1][g[-1]] for g in groups]
        elif level == 1:
            l1 = result[0][0]
            current = [False] + [events[2][l1[g[-1] - 1][-1]] for g in groups]
    return result


def pool(hidden, groups, null):
    return torch.cat([null] + [torch.stack([hidden[i] for i in group]).mean(0, keepdim=True)
                              for group in groups])


def forward(model, data, symbols):
    """Single member reference plus all intermediates for cache state checks."""
    layouts = layout(symbols)
    h0 = model._run_stack(model.drop(model.word_emb(data)), model.stacks["pre"])
    h1 = model._run_stack(model.down_ln1(pool(h0, layouts[0][0], model.null_1)), model.stacks["l1_down"])
    h2 = model._run_stack(model.down_ln2(pool(h1, layouts[1][0], model.null_2)), model.stacks["l2_down"])
    h3 = model._run_stack(model.down_ln3(pool(h2, layouts[2][0], model.null_3)), model.stacks["l3"])
    e2 = model._run_stack(h3[layouts[2][2]] + h2, model.stacks["l2_up"])
    f1 = model._run_stack(e2[layouts[1][2]] + h1, model.stacks["l1_up"])
    g0 = model._run_stack(f1[layouts[0][2]] + h0, model.stacks["post"])
    return model.final_cast(g0), (h0, h1, h2, h3, e2, f1), layouts
