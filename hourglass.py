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

from shortening import downsample, upsample, level_boundaries


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
            w_head_q, w_head_k, w_head_v = self.qkv_net(self.layer_norm(w))
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

    def step(self, w_new, r, r_w_bias, r_r_bias, cache_k, cache_v):
        """Incremental attention for one new position with a KV cache.

        w_new: 1 x B x C (the single new position)
        r:     L x 1 x C positional embeddings for distances L-1..0, where L is
               the cache length *after* appending the new key.
        cache_k / cache_v: L-1 x B x n_head x d_head or None.

        Returns (output 1 x B x C, new_cache_k L x B x n_head x d_head,
        new_cache_v). A single query attends to all L keys, all at positions
        <= the query, so no causal mask is needed.
        """
        assert not self.pre_lnorm
        bsz = w_new.size(1)

        w_heads = self.qkv_net(w_new)
        w_q, w_k, w_v = torch.chunk(w_heads, 3, dim=-1)
        w_q = w_q.view(1, bsz, self.n_head, self.d_head)
        w_k = w_k.view(1, bsz, self.n_head, self.d_head)
        w_v = w_v.view(1, bsz, self.n_head, self.d_head)

        if cache_k is not None:
            w_k = torch.cat([cache_k, w_k], dim=0)
            w_v = torch.cat([cache_v, w_v], dim=0)
        klen = w_k.size(0)

        r_head_k = self.r_net(r).view(klen, self.n_head, self.d_head)

        rw_q = w_q + r_w_bias
        AC = torch.einsum('ibnd,jbnd->bnij', rw_q, w_k)   # B x nh x 1 x L
        rr_q = w_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->bnij', rr_q, r_head_k)  # B x nh x 1 x L
        attn_score = add_and_scale(AC, BD, self.scale)

        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, w_v)
        attn_vec = attn_vec.contiguous().view(1, bsz, self.n_head * self.d_head)

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        output = self.layer_norm(w_new + attn_out)
        return output, w_k, w_v


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

    def step(self, x_new, r, r_w_bias, r_r_bias, cache_k, cache_v):
        output, k, v = self.dec_attn.step(
            x_new, r, r_w_bias, r_r_bias, cache_k, cache_v)
        output = self.pos_ff(output)
        return output, k, v


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
            c3.transpose(0, 1).contiguous())

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

    # ---- incremental KV-cached path (batch size 1) ---------------------
    def _stack_step(self, layers, x_new, caches):
        L = 1 if caches[0] is None else caches[0][0].size(0) + 1
        pos_seq = torch.arange(L - 1, -1, -1.0,
                               device=x_new.device, dtype=x_new.dtype)
        r = self.pos_emb(pos_seq)
        out = x_new
        new_caches = []
        for i, layer in enumerate(layers):
            ck, cv = (None, None) if caches[i] is None else caches[i]
            out, k, v = layer.step(out, r, self.r_w_bias, self.r_r_bias, ck, cv)
            new_caches.append((k, v))
        return out, new_caches

    def init_state(self):
        state = {
            'caches': {n: [None] * len(self.stacks[n]) for n in STACK_NAMES},
            'l1_sum': None, 'l1_cnt': 0,
            'l2_sum': None, 'l2_cnt': 0,
            'l3_sum': None, 'l3_cnt': 0,
            'h3_last': None, 'e2_last': None, 'f1_last': None,
        }
        c = state['caches']
        # Prime the pooled stacks with their processed null group (index 0).
        h1_null, c['l1_down'] = self._stack_step(
            self.stacks['l1_down'], self.down_ln1(self.null_1), c['l1_down'])
        h2_null, c['l2_down'] = self._stack_step(
            self.stacks['l2_down'], self.down_ln2(self.null_2), c['l2_down'])
        h3_null, c['l3'] = self._stack_step(
            self.stacks['l3'], self.down_ln3(self.null_3), c['l3'])
        # Level-2/3 pooling accumulators begin with the processed null.
        state['l2_sum'], state['l2_cnt'] = h1_null.clone(), 1
        state['l3_sum'], state['l3_cnt'] = h2_null.clone(), 1
        state['h3_last'] = h3_null
        # Prime the up-path stacks with the null chain.
        e2_null, c['l2_up'] = self._stack_step(
            self.stacks['l2_up'], state['h3_last'] + h2_null, c['l2_up'])
        state['e2_last'] = e2_null
        f1_null, c['l1_up'] = self._stack_step(
            self.stacks['l1_up'], e2_null + h1_null, c['l1_up'])
        state['f1_last'] = f1_null
        return state

    def step(self, state, token_id, c1, c2, c3):
        """Advance one full-resolution token. c1/c2/c3 are its close events
        (cumulative). Returns logits (1 x 1 x V)."""
        c = state['caches']
        dev = self.r_w_bias.device
        tok = torch.tensor([[int(token_id)]], device=dev)
        x = self.drop(self.word_emb(tok))

        a_t, c['pre'] = self._stack_step(self.stacks['pre'], x, c['pre'])
        res0 = a_t
        state['l1_sum'] = a_t.clone() if state['l1_sum'] is None else state['l1_sum'] + a_t
        state['l1_cnt'] += 1

        if c1:
            g1 = self.down_ln1(state['l1_sum'] / state['l1_cnt'])
            state['l1_sum'], state['l1_cnt'] = None, 0
            h1_new, c['l1_down'] = self._stack_step(self.stacks['l1_down'], g1, c['l1_down'])
            res1 = h1_new
            state['l2_sum'] = h1_new.clone() if state['l2_sum'] is None else state['l2_sum'] + h1_new
            state['l2_cnt'] += 1

            if c2:
                g2 = self.down_ln2(state['l2_sum'] / state['l2_cnt'])
                state['l2_sum'], state['l2_cnt'] = None, 0
                h2_new, c['l2_down'] = self._stack_step(self.stacks['l2_down'], g2, c['l2_down'])
                res2 = h2_new
                state['l3_sum'] = h2_new.clone() if state['l3_sum'] is None else state['l3_sum'] + h2_new
                state['l3_cnt'] += 1

                if c3:
                    g3 = self.down_ln3(state['l3_sum'] / state['l3_cnt'])
                    state['l3_sum'], state['l3_cnt'] = None, 0
                    h3_new, c['l3'] = self._stack_step(self.stacks['l3'], g3, c['l3'])
                    state['h3_last'] = h3_new

                e2_new, c['l2_up'] = self._stack_step(
                    self.stacks['l2_up'], state['h3_last'] + res2, c['l2_up'])
                state['e2_last'] = e2_new

            f1_new, c['l1_up'] = self._stack_step(
                self.stacks['l1_up'], state['e2_last'] + res1, c['l1_up'])
            state['f1_last'] = f1_new

        g0, c['post'] = self._stack_step(
            self.stacks['post'], state['f1_last'] + res0, c['post'])
        return self.final_cast(g0)

    def cached_forward(self, data, c1, c2, c3):
        """Run the cached path over a full given sequence (T x 1). Returns
        logits T x 1 x V. Used to check equivalence with `forward`."""
        assert data.size(1) == 1, "cached path is batch size 1"
        state = self.init_state()
        logits = []
        for t in range(data.size(0)):
            lg = self.step(state, data[t, 0].item(),
                           int(c1[t, 0]), int(c2[t, 0]), int(c3[t, 0]))
            logits.append(lg)
        return torch.cat(logits, dim=0)
