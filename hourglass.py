# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from shortening import check_closes, downsample, upsample, level_boundaries


@torch.jit.script
def add_and_scale(tensor1, tensor2, alpha: float):
    return alpha * (tensor1 + tensor2)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm, activation_function):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)

        return output


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
        self, n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
    ):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()

        del activation_function

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(self.d_model, 3 * n_head * d_head)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)

        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))

        x = x_padded.narrow(2, 1, x_padded.size(2) - 1).view_as(x)

        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask):
        # w is of size: T x B x C
        # r is of size: T x 1 x C
        # biases are of size: (n_head x d_head), we add the same bias to each token
        # attn_mask is of size (q_len x k_len)
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if self.pre_lnorm:
            w_heads = self.qkv_net(self.layer_norm(w))
        else:
            w_heads = self.qkv_net(w)

        r_head_k = self.r_net(r)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)       # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias                                # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->bnij', rw_head_q, w_head_k)      # bsz x n_head x qlen x klen

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->bnij', rr_head_q, r_head_k)       # bsz x n_head x qlen x klen
        BD = self._rel_shift(BD)

        # [bsz x n_head x qlen x klen]
        attn_score = add_and_scale(AC, BD, self.scale)

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))
        else:
            raise NotImplementedError

        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, w_head_v)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

    def step_batched(self, x_in, k_buf, v_buf, fill, dist, rk_table, key_mask,
                     r_w_bias, r_r_bias):
        """Batched incremental attention with preallocated, ragged caches.

        x_in:     1 x B x C new position (query = key = value source).
        k_buf/v_buf: cap x B x n_head x d_head preallocated caches; the new
                  key/value are written in place at row ``fill``.
        dist:     L x B long, each slot's ordinal distance from the query in
                  that sequence's own (padding-free) numbering (L = fill + 1).
        rk_table: cap x n_head x d_head -- ``r_net`` applied once to the
                  sinusoid distance table. ``r_net`` is linear, so
                  ``r_net(table)[d] == r_net(table[d])`` and the per-step
                  relative-position keys are a gather rather than a projection
                  of the whole key history.
        key_mask: B x L bool, True where a slot must NOT be attended (a padding
                  slot that is not a real group for that sequence).

        One query per sequence attends to that sequence's valid keys at their
        per-sequence ordinal distances, reproducing the naive relative attention
        independently for every batch member. Returns output 1 x B x C.
        """
        assert not self.pre_lnorm
        B = x_in.size(1)
        w_q, w_k, w_v = torch.chunk(self.qkv_net(x_in), 3, dim=-1)
        w_q = w_q.view(1, B, self.n_head, self.d_head)
        k_buf[fill] = w_k.reshape(B, self.n_head, self.d_head)
        v_buf[fill] = w_v.reshape(B, self.n_head, self.d_head)

        L = fill + 1
        k = k_buf[:L]                                    # L x B x nh x dh (view)
        v = v_buf[:L]
        r_head_k = rk_table[dist]                        # L x B x nh x dh

        rw_q = w_q + r_w_bias
        AC = torch.einsum('ibnd,jbnd->bnij', rw_q, k)    # B x nh x 1 x L
        rr_q = w_q + r_r_bias
        BD = torch.einsum('ibnd,jbnd->bnij', rr_q, r_head_k)
        attn_score = add_and_scale(AC, BD, self.scale)
        attn_score = attn_score.masked_fill(key_mask[:, None, None, :],
                                            -float('inf'))

        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, v)
        attn_vec = attn_vec.contiguous().view(1, B, self.n_head * self.d_head)

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        return self.layer_norm(x_in + attn_out)


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        pre_lnorm,
        activation_function,
    ):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
        )
        self.pos_ff = PositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm,
            activation_function,
        )

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask)
        output = self.pos_ff(output)

        return output

    def step_batched(self, x_in, k_buf, v_buf, fill, dist, rk_table, key_mask,
                     r_w_bias, r_r_bias):
        output = self.dec_attn.step_batched(
            x_in, k_buf, v_buf, fill, dist, rk_table, key_mask, r_w_bias, r_r_bias)
        return self.pos_ff(output)


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, pre_lnorm, model_config,
                 activation_function, boundaries_type,
                 ):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = nn.Embedding(n_token, d_model)
        self.drop = nn.Dropout(dropout)

        # Relative attention specific parameters
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(
            torch.Tensor(self.n_head, self.d_head).zero_()
        )
        self.r_r_bias = nn.Parameter(
            torch.Tensor(self.n_head, self.d_head).zero_()
        )

        assert pre_lnorm is False, "We didn't use pre_lnorm"

        def create_decoder_layers(n_layers):
            layers = nn.ModuleList([
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    dropatt=dropatt, pre_lnorm=pre_lnorm,
                    activation_function=activation_function)
                for _ in range(n_layers)
            ])

            return layers

        pre_layers, (shortened_layers, ), post_layers = eval(model_config)

        self.boundaries_type = boundaries_type

        if post_layers == 0 and shortened_layers == 0:
            assert boundaries_type == 'none'
            self.layers = nn.ModuleList([
                create_decoder_layers(pre_layers)
            ])
        else:
            self.null_group = nn.Parameter(torch.Tensor(1, 1, d_model).zero_())
            nn.init.normal_(self.null_group)

            self.layers = nn.ModuleList([
                create_decoder_layers(pre_layers),
                create_decoder_layers(shortened_layers),
                create_decoder_layers(post_layers),
            ])

            self.down_ln = nn.LayerNorm(d_model)

        self.final_cast = nn.Linear(d_model, n_token)
        self.crit = torch.nn.CrossEntropyLoss(reduction='none')

    def _forward(self, core_input, layers):
        # Core_input is of size (T x B x C)
        qlen, _, _ = core_input.size()

        dec_attn_mask = torch.triu(
            core_input.new_ones(qlen, qlen), diagonal=1).bool()

        pos_seq = torch.arange(
            qlen - 1, -1, -1.0, device=core_input.device, dtype=core_input.dtype
        )

        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.drop(pos_emb)

        core_out = core_input
        for i, layer in enumerate(layers):
            core_out = layer(
                core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask
            )

        return core_out

    def forward(self,
                data,
                target,
                boundaries_gt):
        """
            data: T x B
            target: T x B
            boundaries_gt: T x B or None
        """
        stats = {}

        # All batches should be of the same length, but last can be shorter
        tgt_len = target.size(0) if target is not None else data.size(0)

        # Token_ids to vector embeddings -> T x B x C
        word_emb = self.word_emb(data)
        hidden = self.drop(word_emb)

        # Extra variables
        loss_boundaries = torch.tensor(0, dtype=data.dtype, device=data.device)
        residual = None

        # Process input with Transformer blocks
        for i in range(len(self.layers)):
            if i == 1:  # Downsampling
                residual = hidden

                # T x B -> B x T
                hard_boundaries = boundaries_gt.float().transpose(0, 1)

                hidden = downsample(
                    boundaries=hard_boundaries,
                    hidden=hidden,
                    null_group=self.null_group,
                )

                hidden = self.down_ln(hidden)

                # Shortening stats
                stats['p_ones'] = (hard_boundaries.sum() / hard_boundaries.numel()).item()
                stats['loss_boundaries'] = loss_boundaries.item()
                stats['shortened_length'] = hidden.size(0)
            elif i == 2:  # Upsampling
                back_hidden = upsample(
                    boundaries=hard_boundaries,
                    shortened_hidden=hidden,
                )

                hidden = back_hidden + residual

            # Out of downsample / upsample -> regular Transformer blocks
            layers = self.layers[i]

            hidden = self._forward(
                core_input=hidden,
                layers=layers,
            )

        # Calculate loss
        hidden = hidden[-tgt_len:]
        logit = self.final_cast(hidden)

        if self.training or target is not None:
            # T x B x C
            assert hidden.size(0) == target.size(0)

            # LM loss
            logit = logit.view(-1, logit.size(-1))
            target = target.view(-1)

            loss = self.crit(logit, target)
            loss = loss.view(tgt_len, -1)

            return loss, stats, loss_boundaries, logit
        else:
            # Generation mode, we return raw logits
            return logit


# Stack order in the three-level hourglass architecture.
STACK_NAMES = ['pre', 'l1_down', 'l2_down', 'l3', 'l2_up', 'l1_up', 'post']
# Stacks that advance once per token; the rest advance once per closed group.
TOKEN_RATE_STACKS = frozenset({'pre', 'post'})


class HourglassLM(nn.Module):
    """Three-level dynamic-pooling transformer.

    Architecture (with residuals joining matching-resolution reps):

        token transformer (pre)
        -> pool L1 -> L1 transformer
        -> pool L2 -> L2 transformer
        -> pool L3 -> L3 transformer
        -> upsample L3 -> L2 transformer (up)   [+ res L2]
        -> upsample L2 -> L1 transformer (up)   [+ res L1]
        -> upsample L1 -> token transformer     [+ res token]
        -> logits

    Two equivalent inference paths are provided: a naive full-recompute
    `forward`, and an incremental KV-cached `step`/`cached_forward`.
    """

    def __init__(self, n_token, n_head, d_model, d_head, d_inner,
                 dropout=0.0, dropatt=0.0, activation_function='gelu',
                 layers=(1, 1, 1, 1, 1, 1, 1)):
        super().__init__()
        assert len(layers) == 7
        self.n_token = n_token
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = nn.Embedding(n_token, d_model)
        self.drop = nn.Dropout(dropout)

        self.pos_emb = PositionalEmbedding(d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(n_head, d_head).zero_())
        self.r_r_bias = nn.Parameter(torch.Tensor(n_head, d_head).zero_())

        def make_layers(n):
            return nn.ModuleList([
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    dropatt=dropatt, pre_lnorm=False,
                    activation_function=activation_function)
                for _ in range(n)
            ])

        self.stacks = nn.ModuleDict(
            {name: make_layers(n) for name, n in zip(STACK_NAMES, layers)})

        # One learned null-group per down level + its post-pool LayerNorm.
        self.null_1 = nn.Parameter(torch.Tensor(1, 1, d_model).normal_())
        self.null_2 = nn.Parameter(torch.Tensor(1, 1, d_model).normal_())
        self.null_3 = nn.Parameter(torch.Tensor(1, 1, d_model).normal_())
        self.down_ln1 = nn.LayerNorm(d_model)
        self.down_ln2 = nn.LayerNorm(d_model)
        self.down_ln3 = nn.LayerNorm(d_model)

        self.final_cast = nn.Linear(d_model, n_token)
        self.crit = nn.CrossEntropyLoss(reduction='none')

    # ---- shared full-sequence stack ------------------------------------
    def _run_stack(self, core_input, layers):
        qlen = core_input.size(0)
        dec_attn_mask = torch.triu(
            core_input.new_ones(qlen, qlen), diagonal=1).bool()
        pos_seq = torch.arange(qlen - 1, -1, -1.0,
                               device=core_input.device, dtype=core_input.dtype)
        pos_emb = self.drop(self.pos_emb(pos_seq))
        out = core_input
        for layer in layers:
            out = layer(out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask)
        return out

    # ---- naive full-recompute forward ----------------------------------
    def forward(self, data, c1, c2, c3, target=None):
        """data, c1, c2, c3: T x B. Returns logits (T x B x V), or (logits,
        loss T x B) when target is given."""
        tgt_len, bsz = data.size(0), data.size(1)
        hidden = self.drop(self.word_emb(data))

        h0 = self._run_stack(hidden, self.stacks['pre'])
        res0 = h0

        bnd1, bnd2, bnd3 = level_boundaries(
            c1.transpose(0, 1).contiguous(),
            c2.transpose(0, 1).contiguous(),
            c3.transpose(0, 1).contiguous(),
            dtype=hidden.dtype)

        h1 = self.down_ln1(downsample(bnd1, h0, self.null_1))
        h1 = self._run_stack(h1, self.stacks['l1_down'])
        res1 = h1

        h2 = self.down_ln2(downsample(bnd2, h1, self.null_2))
        h2 = self._run_stack(h2, self.stacks['l2_down'])
        res2 = h2

        h3 = self.down_ln3(downsample(bnd3, h2, self.null_3))
        h3 = self._run_stack(h3, self.stacks['l3'])

        e2 = self._run_stack(upsample(bnd3, h3) + res2, self.stacks['l2_up'])
        f1 = self._run_stack(upsample(bnd2, e2) + res1, self.stacks['l1_up'])
        g0 = self._run_stack(upsample(bnd1, f1) + res0, self.stacks['post'])

        logit = self.final_cast(g0)

        if target is not None:
            loss = self.crit(logit.view(-1, logit.size(-1)), target.reshape(-1))
            loss = loss.view(tgt_len, bsz)
            return logit, loss
        return logit

    # ---- incremental KV-cached path (batched) --------------------------
    #
    # Every sequence in the batch groups at its own rate, so each shortened
    # stack holds a *ragged* set of real group representations. They share one
    # preallocated cache per layer: on a step where at least one sequence closes
    # a group, all sequences append a slot, real for those that closed and a
    # padding slot for the rest. Each query attends only to its own real slots
    # (via `key_mask`) at its own ordinal distances (via `remb`), which makes the
    # cached attention identical to the naive dense attention for every member.

    def _ensure_cap(self, state, name, need):
        """Grow a stack's preallocated caches (doubling) to hold `need` slots."""
        cur = state['valid'][name].size(0)
        if need <= cur:
            return
        newcap = max(need, cur * 2)
        bsz, dev = state['bsz'], state['valid'][name].device
        for store in ('k', 'v'):
            for i, old in enumerate(state[store][name]):
                buf = torch.zeros(newcap, bsz, self.n_head, self.d_head,
                                  device=dev, dtype=state['dtype'])
                buf[:cur] = old
                state[store][name][i] = buf
        ov = state['valid'][name]
        nv = torch.zeros(newcap, bsz, dtype=torch.bool, device=dev)
        nv[:cur] = ov
        state['valid'][name] = nv

    def _geom(self, valid_new):
        """Per-sequence relative geometry for a batched step.

        valid_new: L x B bool (which of the L slots are real for each sequence,
        including the just-appended slot). Returns each slot's ordinal distance
        from the query in that sequence's own padding-free numbering (L x B) and
        the attention key mask (B x L, True = do not attend).
        """
        L = valid_new.size(0)
        ordv = valid_new.long().cumsum(0)                 # L x B: #valid up to slot
        dist = (ordv[-1].unsqueeze(0) - ordv).clamp_(0, L - 1)   # L x B
        key_mask = ~valid_new.transpose(0, 1).contiguous()  # B x L
        key_mask[:, -1] = False   # a query always sees its own new slot (no NaN)
        return dist, key_mask

    def _ensure_pos(self, state, need):
        """(Re)build the cached relative-position keys to cover `need` distances.

        The sinusoid distance table and each attention's ``r_net`` projection of
        it depend only on the distance, so they are built once per decode and
        gathered per step instead of being recomputed over the whole key history
        at every step. This assumes the weights are frozen for the lifetime of
        the state, which decoding already requires.
        """
        if state['pos_cap'] >= need:
            return
        cap = max(need, 2 * state['pos_cap'], 8)
        pos = torch.arange(cap, device=state['device'], dtype=state['dtype'])
        table = self.pos_emb(pos).squeeze(1)              # cap x C
        with torch.no_grad():
            for name in STACK_NAMES:
                state['rk'][name] = [
                    layer.dec_attn.r_net(table).view(cap, self.n_head, self.d_head)
                    for layer in self.stacks[name]
                ]
        state['pos_cap'] = cap

    def _stack_step_batched(self, state, name, x_new, active):
        """Advance one stack by a single (batched) position.

        active: B bool -- which sequences get a *real* new slot; the rest append
        a padding slot. A zero-layer stack is the identity (no cache)."""
        layers = self.stacks[name]
        if len(layers) == 0:
            return x_new
        fill = state['fill'][name]
        self._ensure_cap(state, name, fill + 1)
        self._ensure_pos(state, fill + 1)
        state['valid'][name][fill] = active
        dist, key_mask = self._geom(state['valid'][name][:fill + 1])
        out = x_new
        for i, layer in enumerate(layers):
            out = layer.step_batched(
                out, state['k'][name][i], state['v'][name][i], fill, dist,
                state['rk'][name][i], key_mask, self.r_w_bias, self.r_r_bias)
        state['fill'][name] = fill + 1
        return out

    def init_state_batched(self, bsz, max_len=None, device=None):
        """Fresh batched cache state.

        `max_len` preallocates the token-rate stacks (`pre`/`post`) exactly:
        they advance once per token, so `max_len + 1` slots is both necessary
        and sufficient. The pooled stacks advance only when a group closes, at a
        data-dependent compression ratio, so preallocating them to `max_len`
        wastes most of the buffer (a level-3 stack typically fills a few percent
        of it). They start small and grow by doubling instead, which is
        bit-identical to preallocating and adapts to the actual grouping rate.

        The cached path is inference-only: keys and values are written into the
        preallocated buffers in place, which autograd cannot track. Train with
        the naive `forward`, which is also the correctness reference.
        """
        dev = device or self.r_w_bias.device
        dt = self.r_w_bias.dtype
        token_cap = (max_len + 1) if max_len is not None else 8
        pooled_cap = min(token_cap, 16)
        zl = lambda: torch.zeros(bsz, dtype=torch.long, device=dev)
        state = {
            'bsz': bsz, 'device': dev, 'dtype': dt,
            'k': {}, 'v': {}, 'valid': {}, 'fill': {},
            'rk': {name: [] for name in STACK_NAMES}, 'pos_cap': 0,
            'l1_sum': None, 'l1_cnt': zl(),
            'l2_sum': None, 'l2_cnt': zl(),
            'l3_sum': None, 'l3_cnt': zl(),
            'h3_last': None, 'e2_last': None, 'f1_last': None,
        }
        def empty_cache(cap):
            return torch.zeros(cap, bsz, self.n_head, self.d_head,
                               device=dev, dtype=dt)

        for name in STACK_NAMES:
            nl = len(self.stacks[name])
            cap = token_cap if name in TOKEN_RATE_STACKS else pooled_cap
            state['k'][name] = [empty_cache(cap) for _ in range(nl)]
            state['v'][name] = [empty_cache(cap) for _ in range(nl)]
            state['valid'][name] = torch.zeros(cap, bsz, dtype=torch.bool, device=dev)
            state['fill'][name] = 0

        active = torch.ones(bsz, dtype=torch.bool, device=dev)
        rep = lambda p: p.expand(1, bsz, self.d_model)   # 1x1xC param -> 1xBxC
        # Prime the pooled + up stacks with their processed null group (slot 0).
        h1_null = self._stack_step_batched(state, 'l1_down', self.down_ln1(rep(self.null_1)), active)
        h2_null = self._stack_step_batched(state, 'l2_down', self.down_ln2(rep(self.null_2)), active)
        h3_null = self._stack_step_batched(state, 'l3', self.down_ln3(rep(self.null_3)), active)
        ones = torch.ones(bsz, dtype=torch.long, device=dev)
        state['l2_sum'], state['l2_cnt'] = h1_null.clone(), ones.clone()
        state['l3_sum'], state['l3_cnt'] = h2_null.clone(), ones.clone()
        state['h3_last'] = h3_null
        e2_null = self._stack_step_batched(state, 'l2_up', state['h3_last'] + h2_null, active)
        state['e2_last'] = e2_null
        f1_null = self._stack_step_batched(state, 'l1_up', e2_null + h1_null, active)
        state['f1_last'] = f1_null
        return state

    def step_batched(self, state, tokens, c1, c2, c3, active=None):
        """Advance one full-resolution token for the whole batch.

        tokens, c1, c2, c3: B (token ids and their cumulative close events).
        active: B bool -- sequences still decoding (finished ones are frozen and
        contribute nothing). Returns logits 1 x B x V."""
        B = state['bsz']
        dev = self.r_w_bias.device
        dt = state['dtype']
        check_closes(c1, c2, c3)
        if active is None:
            active = torch.ones(B, dtype=torch.bool, device=dev)
        am = active.view(1, B, 1).to(dt)
        m1 = c1.bool() & active
        m2 = c2.bool() & active
        m3 = c3.bool() & active

        x = self.drop(self.word_emb(tokens.view(1, B)))
        a_t = self._stack_step_batched(state, 'pre', x, active)
        res0 = a_t
        add = a_t * am
        state['l1_sum'] = add if state['l1_sum'] is None else state['l1_sum'] + add
        state['l1_cnt'] = state['l1_cnt'] + active.long()

        if bool(m1.any()):
            cnt = state['l1_cnt'].clamp(min=1).view(1, B, 1).to(dt)
            g1 = self.down_ln1(state['l1_sum'] / cnt)
            r1 = m1.view(1, B, 1)
            state['l1_sum'] = torch.where(r1, torch.zeros_like(state['l1_sum']), state['l1_sum'])
            state['l1_cnt'] = torch.where(m1, torch.zeros_like(state['l1_cnt']), state['l1_cnt'])
            h1 = self._stack_step_batched(state, 'l1_down', g1, m1)
            res1 = h1
            add1 = h1 * r1.to(dt)
            state['l2_sum'] = state['l2_sum'] + add1
            state['l2_cnt'] = state['l2_cnt'] + m1.long()

            if bool(m2.any()):
                cnt2 = state['l2_cnt'].clamp(min=1).view(1, B, 1).to(dt)
                g2 = self.down_ln2(state['l2_sum'] / cnt2)
                r2 = m2.view(1, B, 1)
                state['l2_sum'] = torch.where(r2, torch.zeros_like(state['l2_sum']), state['l2_sum'])
                state['l2_cnt'] = torch.where(m2, torch.zeros_like(state['l2_cnt']), state['l2_cnt'])
                h2 = self._stack_step_batched(state, 'l2_down', g2, m2)
                res2 = h2
                add2 = h2 * r2.to(dt)
                state['l3_sum'] = state['l3_sum'] + add2
                state['l3_cnt'] = state['l3_cnt'] + m2.long()

                if bool(m3.any()):
                    cnt3 = state['l3_cnt'].clamp(min=1).view(1, B, 1).to(dt)
                    g3 = self.down_ln3(state['l3_sum'] / cnt3)
                    r3 = m3.view(1, B, 1)
                    state['l3_sum'] = torch.where(r3, torch.zeros_like(state['l3_sum']), state['l3_sum'])
                    state['l3_cnt'] = torch.where(m3, torch.zeros_like(state['l3_cnt']), state['l3_cnt'])
                    h3 = self._stack_step_batched(state, 'l3', g3, m3)
                    state['h3_last'] = torch.where(r3, h3, state['h3_last'])

                e2 = self._stack_step_batched(state, 'l2_up', state['h3_last'] + res2, m2)
                state['e2_last'] = torch.where(r2, e2, state['e2_last'])

            f1 = self._stack_step_batched(state, 'l1_up', state['e2_last'] + res1, m1)
            state['f1_last'] = torch.where(r1, f1, state['f1_last'])

        g0 = self._stack_step_batched(state, 'post', state['f1_last'] + res0, active)
        return self.final_cast(g0)

    def cached_forward_batched(self, data, c1, c2, c3):
        """Run the batched cached path over a full T x B sequence. Returns logits
        T x B x V. Used to check equivalence with the naive `forward`."""
        T, B = data.size(0), data.size(1)
        state = self.init_state_batched(B, max_len=T, device=data.device)
        logits = [self.step_batched(state, data[t], c1[t], c2[t], c3[t])
                  for t in range(T)]
        return torch.cat(logits, dim=0)

    # ---- batch-size-1 convenience wrappers (used by the streaming decoders) --
    def init_state(self):
        return self.init_state_batched(1)

    def step(self, state, token_id, c1, c2, c3):
        """Advance one token for a batch-size-1 state. Returns logits 1 x 1 x V."""
        dev = self.r_w_bias.device
        t = lambda v: torch.tensor([int(v)], device=dev)
        return self.step_batched(state, t(token_id), t(c1), t(c2), t(c3))

    def cached_forward(self, data, c1, c2, c3):
        """Batched cached path restricted to T x 1 (kept for existing callers)."""
        return self.cached_forward_batched(data, c1, c2, c3)
