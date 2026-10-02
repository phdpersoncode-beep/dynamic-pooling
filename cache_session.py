"""CPU reference session and restart format for validation, not a serving API.

Weights remain external. Restoration requires identical weights, rules, dtype,
PyTorch version and backend. Sampling and cross-device migration are out of scope.
"""
import hashlib
import inspect
import os
from pathlib import Path

import torch
from hierarchy_v2 import CONTRACT, fingerprint


def weights_hash(model):
    digest = hashlib.sha256(repr(model).encode())
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def identity(model, tokenizer):
    if model.training or next(model.parameters()).device.type != "cpu":
        raise ValueError("reference sessions require eval mode on CPU")
    if tokenizer.group_rule is not None and tokenizer.group_rule_name is None:
        raise ValueError("register a versioned rule before saving a session")
    rule_source = inspect.getsource(tokenizer.group_rule) if tokenizer.group_rule else "literal-cumulative-v1"
    return {"format": 1, "contract": CONTRACT, "weights": weights_hash(model),
            "tokenizer": tokenizer.to_meta(), "rule_hash": fingerprint(rule_source),
            "dtype": str(next(model.parameters()).dtype), "torch": str(torch.__version__),
            "mkldnn": torch.backends.mkldnn.enabled,
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "sampling": False}


class CacheSession:
    def __init__(self, model, tokenizer, batch_size=1):
        self.metadata = identity(model, tokenizer)
        self.model, self.tokenizer = model, tokenizer
        with torch.no_grad():
            self.state = model.init_state_batched(batch_size)
        self.group_states = [tokenizer.init_group_state() for _ in range(batch_size)]

    @torch.no_grad()
    def consume(self, data):
        if data.ndim != 2 or data.shape[1] != self.state["bsz"] or len(data) == 0:
            raise ValueError("expected nonempty time x batch tokens")
        if data.dtype != torch.long or data.device.type != "cpu" or not bool(((0 <= data) & (data < len(self.tokenizer))).all()):
            raise ValueError("expected CPU int64 tokens in the vocabulary")
        outputs = []
        for tokens in data:
            events = torch.tensor([self.tokenizer.group(token, group) for token, group in
                                   zip(tokens.tolist(), self.group_states)]).T
            outputs.append(self.model.step_batched(self.state, tokens, *events))
        return torch.cat(outputs)

    def save(self, path):
        if identity(self.model, self.tokenizer) != self.metadata:
            raise ValueError("model/rule/backend changed during the session")
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"metadata": self.metadata, "cache": self.state,
                    "group_states": self.group_states}, temporary)
        os.replace(temporary, path)

    @classmethod
    def restore(cls, path, model, tokenizer):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        expected = identity(model, tokenizer)
        if saved["metadata"] != expected:
            raise ValueError("snapshot weights/rule/contract/backend mismatch")
        result = cls.__new__(cls)
        result.model, result.tokenizer, result.metadata = model, tokenizer, expected
        result.state, result.group_states = saved["cache"], saved["group_states"]
        return result
