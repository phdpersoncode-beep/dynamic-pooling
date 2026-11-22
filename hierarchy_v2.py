"""Versioned, nonempty trees. Pooling semantics are unchanged from v1."""
import hashlib
import json
import random
from pathlib import Path

CONTRACT = {
    "version": 2, "grammar": "nonempty-tree-v2", "rule": "literal-cumulative-v1",
    "pooling": "original-inclusive-equal-child-v1", "eos": "stop-without-flush",
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def linearize(tree):
    """tree[top][parent][leaf] is a nonempty list of payload symbol strings."""
    if not tree:
        raise ValueError("a tree needs a top-level group")
    symbols = ["SOS"]
    for top in tree:
        if not top:
            raise ValueError("a top-level group needs children")
        for p, parent in enumerate(top):
            if not parent:
                raise ValueError("a parent needs leaves")
            for i, leaf in enumerate(parent):
                if not leaf or any(not is_payload(x) for x in leaf):
                    raise ValueError("a leaf needs payload symbols")
                symbols.extend(leaf)
                symbols.append("b1" if i + 1 < len(parent) else
                               "b2" if p + 1 < len(top) else "b3")
    return symbols + ["EOS"]


def is_payload(symbol):
    return isinstance(symbol, str) and symbol.startswith("x") and symbol[1:].isdigit() and 0 <= int(symbol[1:]) < 256


def validate(symbols, complete=True):
    """Validate task grammar, independently of the permissive model API."""
    if not symbols or symbols[0] != "SOS":
        raise ValueError("expected initial SOS")
    payload = 0
    for i, symbol in enumerate(symbols[1:], 1):
        if is_payload(symbol):
            payload += 1
        elif symbol in ("b1", "b2", "b3"):
            if payload == 0:
                raise ValueError("empty leaf")
            payload = 0
        elif symbol == "EOS":
            if i != len(symbols) - 1 or symbols[i - 1] != "b3":
                raise ValueError("EOS must follow final b3")
        else:
            raise ValueError(f"unexpected symbol {symbol!r}")
    if complete and (len(symbols) < 4 or symbols[-1] != "EOS"):
        raise ValueError("complete examples must end in b3 EOS")
    return True


def random_tree(seed, top_count=2, max_parents=3, max_leaves=3, max_payload=5, n_x=16):
    rng = random.Random(seed)
    return [[[ [f"x{rng.randrange(n_x)}" for _ in range(rng.randint(1, max_payload))]
               for _ in range(rng.randint(1, max_leaves))]
             for _ in range(rng.randint(1, max_parents))] for _ in range(top_count)]


def task_example(task, seed, motif_length=2, distractors=1, n_x=16, distractor_length=None):
    """Fixed child counts make recall timing observable without new cue tokens.

    Returns symbols, prediction target roles, and source motif identity.
    Roles align to symbol positions (not shifted input positions).
    """
    rng = random.Random(seed)
    motif = lambda: [f"x{rng.randrange(n_x)}" for _ in range(motif_length)]
    noise = lambda: [f"x{rng.randrange(n_x)}" for _ in range(distractor_length or motif_length)]
    source = motif()
    if task == "l1_repeat":
        source = source[:2]
        tree = [[[[symbol] * (motif_length + 2) for symbol in source]]]
    elif task == "l2_copy":
        tree = [[[source] + [noise() for _ in range(distractors)] + [source.copy()]]]
    elif task == "l3_copy":
        source = [source, motif()]
        tree = [[source] + [[noise(), noise()] for _ in range(distractors)] +
                [[leaf.copy() for leaf in source]]]
    else:
        raise ValueError(f"unknown task {task}")
    symbols = linearize(tree)
    roles = ["boundary" if s.startswith("b") else "eos" if s == "EOS" else "source"
             for s in symbols]
    if task == "l1_repeat":
        for i in range(2, len(symbols) - 2):
            if is_payload(symbols[i]) and is_payload(symbols[i - 1]):
                roles[i] = "copy"
    else:
        begin = max(i for i, s in enumerate(symbols) if s == ("b1" if task == "l2_copy" else "b2")) + 1
        for i in range(begin, len(symbols) - 2):
            if is_payload(symbols[i]):
                roles[i] = "copy"
    return {"symbols": symbols, "roles": roles, "source_id": fingerprint(source),
            "tree": tree, "seed": seed, "task": task}


def save_dataset(path, examples, tokenizer_meta):
    for example in examples:
        validate(example["symbols"])
    payload = {"contract": CONTRACT, "tokenizer": tokenizer_meta, "examples": examples}
    payload["sha256"] = fingerprint(payload)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def load_dataset(path):
    payload = json.loads(Path(path).read_text())
    digest = payload.pop("sha256")
    if fingerprint(payload) != digest or payload["contract"] != CONTRACT:
        raise ValueError("dataset hash or semantic contract mismatch")
    if payload["tokenizer"].get("group_rule") is not None:
        raise ValueError("v2 task grammar requires the literal boundary rule")
    for example in payload["examples"]:
        validate(example["symbols"])
    return payload
