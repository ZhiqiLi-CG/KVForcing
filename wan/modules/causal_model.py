from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
# ======== ours ========
import torch.utils.checkpoint
from typing import NamedTuple
from wan.modules import attention as _attention_backend
from wan.modules.model import WanT2VCrossAttention
# ==== ours end ====
import math
import torch.distributed as dist

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention,
    dynamic=False,
    mode="default"
)



def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def causal_rope_apply_frames(x, grid_sizes, freqs, frame_indices):
    """RoPE variant where every latent frame is given an *explicit* temporal
    index (instead of a contiguous ``start_frame:start_frame+f`` range).

    This is used by the streaming long-video KV cache: keys/values are stored
    *before* RoPE is applied, and on every step we re-apply RoPE with relative
    indices ``frame_indices`` so the temporal position never leaves the model's
    trained range (e.g. 0-20).

    Args:
        x (Tensor): Shape [B, L, n, d], with ``L = f*h*w`` for each sample.
        grid_sizes (Tensor): Shape [B, 3] with (f, h, w) per sample.
        frame_indices (LongTensor): Shape [f], the temporal RoPE index for each
            of the ``f`` latent frames.
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    frame_indices = frame_indices.to(freqs[0].device)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][frame_indices].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


# ======== ours ========
# ======== ours (Memory-LoRA distance gate) ========
# memory_lora_distance_gate (top-level config key): the Memory-LoRA contribution of a cached frame depends on its
# distance d = clamp(pos_query - pos_key, 0, 20) in temporal RoPE positions to the frame being computed (d = 0 for
# a frame's own K/V). Per layer, separately for K and V (z = down(x), u = up(z) * scale = the ungated dW x):
#   scalar   dW(d) x = g_d * u                 g in R^{21}           init 1
#   rank     dW(d) x = up(g_d (.) z) * scale    g in R^{21 x rank}    init 1
#   channel  dW(d) x = g_d (.) u                g in R^{21 x dim}     init 1
#   input    dW(d) x = up(down(x + b_d)) * scale = up(z + down(b_d)) * scale (down is linear, no bias)
#                                              b in R^{21 x dim}     init 0
#   mix      every memory write keeps BOTH versions of its K/V, K^L = RoPE(norm_k(W_k x + dW_k x)), K^0 = RoPE(norm_k(W_k x)),
#            V^L = W_v x + dW_v x, V^0 = W_v x; a reader at distance d uses K = g_d K^L + (1 - g_d) K^0 (after norm_k and
#            RoPE) and V = V^0 + g'_d dW_v x (= g'_d V^L + (1 - g'_d) V^0), g = sigmoid(theta)
#                                              theta in R^{21}       init 0 (g = 0.5; NOT the ungated model, which is g = 1)
# At init the variants scalar, rank, channel, input are the ungated Memory-LoRA. "none" (default) builds no gate and leaves every code path as
# it was. A gated page is read at a different distance by every later frame, so a gated layer keeps the parts of a
# memory write that do not depend on the reader (W_k x and W_v x before norm_k / RoPE, and z_k, z_v) and builds
# K = RoPE(norm_k(W_k x + dW_k(d) x)), V = W_v x + dW_v(d) x when it reads them (CausalWanSelfAttention._gated_kv).
MEMORY_LORA_GATE_DISTANCES = 21          # d = 0..20, the 21-frame training clip
MEMORY_LORA_DISTANCE_GATES = ("scalar", "rank", "channel", "input", "mix")


def memory_lora_distance_gate_mode(value):
    """The config value of memory_lora_distance_gate as the model takes it: None for no gate."""
    if value is None or value is False or str(value).lower() in ("", "none"):
        return None
    if value not in MEMORY_LORA_DISTANCE_GATES:
        raise ValueError(f"memory_lora_distance_gate must be none or one of {MEMORY_LORA_DISTANCE_GATES}; got {value!r}")
    return value


def memory_lora_distance_gate_from_config(config):
    return memory_lora_distance_gate_mode(getattr(config, "memory_lora_distance_gate", None))
# ==== ours end ====


class MemoryLoRA(nn.Module):
    """Low-rank update to a projection weight: dW h = up(down(h)) * scale.

    Memory-LoRA gives the memory WRITE its own K/V projection (W + dW) while
    generation keeps W. `up` is zero-initialised, so dW starts at exactly zero
    and the pretrained model is reproduced at step 0.
    """

    def __init__(self, dim, rank=64, alpha=None,
                 # ======== ours (distance gate) ========
                 distance_gate=None
                 # ==== ours end ====
                 ):
        super().__init__()
        self.rank = rank
        self.scale = (alpha if alpha is not None else rank) / rank
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        # ======== ours (distance gate) ========
        self.gate_mode = memory_lora_distance_gate_mode(distance_gate)
        if self.gate_mode is not None:
            n = MEMORY_LORA_GATE_DISTANCES
            shape = {"scalar": (n,), "rank": (n, rank), "channel": (n, dim), "input": (n, dim), "mix": (n,)}[self.gate_mode]
            self.distance_gate = nn.Parameter(torch.empty(shape))
        # ==== ours end ====
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.down.weight, std=1.0 / self.rank)
        nn.init.zeros_(self.up.weight)
        # ======== ours (distance gate) ========
        if self.gate_mode is not None:
            self.reset_distance_gate()
        # ==== ours end ====

    def forward(self, x):
        return self.up(self.down(x)) * self.scale

    # ======== ours (distance gate) ========
    @torch.no_grad()
    def reset_distance_gate(self):
        """The init: g = 1 (scalar, rank, channel), b = 0 (input) -- the identity; theta = 0 (mix: g = 0.5)."""
        self.distance_gate.fill_(0.0 if self.gate_mode in ("input", "mix") else 1.0)

    def mix_weight(self, dist):
        """mix: g_d = sigmoid(theta_d) of the distances `dist` (LongTensor [m])."""
        return torch.sigmoid(self.distance_gate.index_select(0, dist))

    def gated(self, z, dist):
        """dW(d) x of m frames from z = down(x) [B, m, L, rank], frame j at distance dist[j] (LongTensor [m]).
        Returns [B, m, L, dim]. With the identity gate it is forward(x), op for op (a factor 1 or a term 0 aside)."""
        m = dist.numel()
        if self.gate_mode == "input":
            # down(b_d) of all 21 distances in ONE fixed-shape GEMM, then the rows of `dist`: a frame's down(b_d)
            # does not depend on how many / which distances a forward reads (a GEMM over another number of rows may
            # reduce in another order, and training and inference read different sets of frames)
            return self.up(z + self.down(self.distance_gate).index_select(0, dist).view(1, m, 1, -1)) * self.scale
        g = self.distance_gate.index_select(0, dist)
        if self.gate_mode == "scalar":
            return self.up(z) * self.scale * g.view(1, m, 1, 1)
        if self.gate_mode == "channel":
            return self.up(z) * self.scale * g.view(1, m, 1, -1)
        return self.up(z * g.view(1, m, 1, -1)) * self.scale        # rank
    # ==== ours end ====
# ==== ours end ====


# ======== ours ========
def _attn_dtype(dtype):
    """The dtype attention() hands flash-attn for inputs of `dtype`."""
    return dtype if dtype in (torch.float16, torch.bfloat16) else torch.bfloat16


class _PageTable:
    """One layer's contiguous page table: K and V buffers [B, max_frames * L, n, d]
    in which the page of frame i occupies rows [i * L, (i + 1) * L). Allocated on
    first use and only ever appended to, in the dtype flash-attn computes in.
    It also keeps the layer's cross-attention K/V of the prompt for the forwards
    without gradient (the inference path's crossattn_cache, per rollout).
    """

    def __init__(self, max_frames):
        self.max_frames = max_frames
        self.k = self.v = None
        self.context_kv = None

    def put(self, slot, k, v):
        """Writes k, v [B, L, n, d] into the slot of frame `slot` and returns the
        slot's views."""
        L = k.shape[1]
        if self.k is None:
            shape = (k.shape[0], self.max_frames * L, *k.shape[2:])
            self.k = k.new_empty(shape, dtype=_attn_dtype(k.dtype))
            self.v = v.new_empty(shape, dtype=_attn_dtype(v.dtype))
        k_slot, v_slot = self.k[:, slot * L:(slot + 1) * L], self.v[:, slot * L:(slot + 1) * L]
        if k.data_ptr() != k_slot.data_ptr():       # else k is this slot already
            k_slot.copy_(k)
        if v.data_ptr() != v_slot.data_ptr():
            v_slot.copy_(v)
        return k_slot, v_slot


class _PageHandle(torch.autograd.Function):
    """A page's place in the autograd graph, without its data: an expanded
    scalar of the page's shape and dtype, through which the page's gradient
    passes unchanged. The data is in the page table."""

    @staticmethod
    def forward(ctx, page):
        return page.new_zeros(()).expand(page.shape)

    @staticmethod
    def backward(ctx, grad):
        return grad


class PageRead(NamedTuple):
    """One layer's history as a forward is handed it: the layer's page table,
    the number of pages in it, and the handles of those pages that carry a
    graph."""
    table: _PageTable
    length: int
    k_handles: tuple
    v_handles: tuple


class HPageRead(NamedTuple):
    """One layer's history for a forward WITH gradient under page_mode "h": the
    h pages, each a writer's self-attention input carrying its graph. The
    history K/V are rebuilt from them (_rebuild_history_kv). The page table is
    not referenced, so nothing keeps it past the rollout."""
    pages: tuple


class PersistentKVStore:
    """Immutable K/V pages written by the KV Forcing rollout.

    A page is the (post-RoPE K, V) one frame contributes to one layer. The
    autoregressive KV cache overwrites a slot in place; a page is never
    modified, so it can carry a graph, and a checkpointed frame recomputes
    against exactly the history it saw at forward time.

    The pages of a layer are the slots of one contiguous page table
    (_PageTable), written once, when the page is appended, in the dtype
    attention() converts K/V to anyway. A page that carries a graph keeps its
    place in it through a handle (_PageHandle).

    A forward with gradient concatenates [history slots | own K/V] and routes
    the history's gradient to the handles (_PageCat). A forward WITHOUT
    gradient (the rollout's other denoising steps, and every forward of DMD's
    critic step) reads the history in place: frame i writes its own K/V into
    slot i, which is scratch until its page is appended, and attends to
    table[:, :(i + 1) * L]; a frame beyond the table's last slot (the last
    frame, which writes no page) concatenates like a forward with gradient.
    Either way flash-attn is given bitwise the K/V it would be given if every
    page were its own tensor. Slots 0..i-1 never change once written, so the
    recomputation of a checkpointed frame reads the history it saw at forward
    time.

    page_mode "h" (default "kv") halves what a generator step keeps for its
    backward pass. The page that carries the graph is then the writer's
    self-attention input h of that layer (dim values per token instead of
    2 * dim for K and V), and so is its gradient. A forward with gradient
    rebuilds the history K/V from the h pages with the writer's own ops inside
    its checkpoint region (_rebuild_history_kv), so they exist only while that
    region is computed or recomputed. The page table still receives every
    writer's K/V, detached, for the forwards without gradient; nothing with a
    graph refers to it, so it is freed with the store, before the backward
    pass. Same function, different order of additions in the backward: the
    rebuilt K/V are bitwise the writers' in bf16, the gradient agrees to
    rounding (see docs/kv_forcing.md).
    """

    def __init__(self, num_layers, max_frames, page_mode="kv"):
        if page_mode not in ("kv", "h"):
            raise ValueError(f"page_mode must be 'kv' or 'h'. Got {page_mode!r}.")
        self.num_layers = num_layers
        self.page_mode = page_mode
        self.tables = [_PageTable(max_frames) for _ in range(num_layers)]
        self.k_handles = [[] for _ in range(num_layers)]
        self.v_handles = [[] for _ in range(num_layers)]
        self.h_pages = [[] for _ in range(num_layers)]
        self.length = 0

    def __len__(self):
        return self.length

    def append_all_layers(self, flat_pages):
        """flat_pages is the writer's flattened output: (k0, v0, k1, v1, ...),
        or, from a writer with gradient under page_mode "h",
        (k0, v0, h0, k1, v1, h1, ...) with k, v detached."""
        width = len(flat_pages) // self.num_layers
        for layer_idx in range(self.num_layers):
            k, v = flat_pages[width * layer_idx], flat_pages[width * layer_idx + 1]
            with torch.no_grad():
                self.tables[layer_idx].put(self.length, k, v)
            if width == 3:
                self.h_pages[layer_idx].append(flat_pages[width * layer_idx + 2])
            elif k.requires_grad:
                self.k_handles[layer_idx].append(_PageHandle.apply(k))
                self.v_handles[layer_idx].append(_PageHandle.apply(v))
        self.length += 1

    def snapshot_all_layers(self):
        """The histories of a forward that carries gradient when the caller
        allows it: a PageRead per layer, or, under page_mode "h" with grad
        enabled, an HPageRead per layer. Appending later cannot change a
        snapshot."""
        if self.page_mode == "h" and torch.is_grad_enabled():
            return tuple(HPageRead(tuple(self.h_pages[l])) for l in range(self.num_layers))
        return tuple(PageRead(self.tables[l], self.length, tuple(self.k_handles[l]), tuple(self.v_handles[l]))
                     for l in range(self.num_layers))

    def table_all_layers(self):
        """The histories of a forward WITHOUT gradient under page_mode "h": the
        page table alone."""
        return tuple(PageRead(self.tables[l], self.length, (), ()) for l in range(self.num_layers))

    def end_rollout(self):
        """Drops the prompt K/V the forwards without gradient shared. The pages
        stay: the graph holds their tables until the backward pass is done."""
        for table in self.tables:
            table.context_kv = None


class _PageCat(torch.autograd.Function):
    """[history | own] for a forward with gradient. `history` is the table's
    view of the pages, in the dtype flash-attn computes in; `own` is cast to
    it. The backward hands each page's handle its slice of the gradient, cast
    back and CONTIGUOUS: views into the one big dK/dV buffer of the whole
    history, which is what cat's own backward returns, would be stored as a
    page's first gradient contribution and pin the entire buffer (30 layers x
    K,V x the whole history) until the page is consumed. The copies free the
    buffer at once. Numerically identical to cat, convert, and its backward.
    """

    @staticmethod
    def forward(ctx, history, own, *handles):
        ctx.sizes = [h.shape[1] for h in handles]
        ctx.split = history.shape[1]
        ctx.dtype = own.dtype
        return torch.cat([history, own.to(history.dtype)], dim=1)

    @staticmethod
    def backward(ctx, grad):
        def owned(g):
            return g.to(ctx.dtype, copy=True, memory_format=torch.contiguous_format)
        pages = grad[:, :ctx.split].split(ctx.sizes, dim=1) if ctx.sizes else ()
        return (None, owned(grad[:, ctx.split:]), *(owned(g) for g in pages))


class _PageCatPair(torch.autograd.Function):
    """_PageCat for the fused frame forward (CausalWanModel.forward_fused_frame),
    whose batch is [noisy; write]: both halves read the same history slots and
    each appends its own K (or V). Returns [history | own_noisy] stacked on
    [history | own_write]. The backward adds the two halves' history gradient
    page by page, after casting each back, which is the sum autograd makes of
    the two separate forwards' contributions; every gradient it hands out is a
    fresh contiguous tensor, for the reason _PageCat gives.
    """

    @staticmethod
    def forward(ctx, history, own_noisy, own_write, *handles):
        ctx.sizes = [h.shape[1] for h in handles]
        ctx.split = history.shape[1]
        ctx.dtype = own_noisy.dtype
        b = own_noisy.shape[0]
        out = history.new_empty((2 * b, ctx.split + own_noisy.shape[1], *history.shape[2:]))
        out[:b, :ctx.split] = history
        out[b:, :ctx.split] = history
        out[:b, ctx.split:] = own_noisy
        out[b:, ctx.split:] = own_write
        return out

    @staticmethod
    def backward(ctx, grad):
        b, page_grads, a = grad.shape[0] // 2, [], 0
        for size in ctx.sizes:
            page_grads.append(grad[:b, a:a + size].to(ctx.dtype) + grad[b:, a:a + size].to(ctx.dtype))
            a += size

        def owned(g):
            return g.to(ctx.dtype, copy=True, memory_format=torch.contiguous_format)
        return (None, owned(grad[:b, ctx.split:]), owned(grad[b:, ctx.split:]), *page_grads)


class _HPageCat(torch.autograd.Function):
    """torch.cat over the h pages of a layer whose backward hands each page a
    CONTIGUOUS gradient instead of a view into the whole history's (see
    _PageCat). Numerically identical to cat."""

    @staticmethod
    def forward(ctx, *pages):
        ctx.sizes = [p.shape[1] for p in pages]
        return torch.cat(pages, dim=1)

    @staticmethod
    def backward(ctx, grad):
        return tuple(g.clone(memory_format=torch.contiguous_format) for g in grad.split(ctx.sizes, dim=1))


def _rebuild_history_kv(attn, pages, rope_hist):
    """The history K/V of one layer from its h pages, with the ops the writers
    ran on the same h, in the same order: K = RoPE(norm_k(W_k h + dW_k h)),
    V = W_v h + dW_v h (a page is a memory write, so the Memory-LoRA applies to
    every page). One GEMM over the concatenated history instead of one per
    page, and RoPE over frames 0..m-1 at once (`rope_hist`); every output row
    depends on its own input row only."""
    b, s = pages[0].shape[:2]
    m, n, d = len(pages), attn.num_heads, attn.head_dim
    hh = _HPageCat.apply(*pages)
    kk, vv = attn.k(hh), attn.v(hh)
    if attn.memory_lora:
        kk, vv = kk + attn.k_memory_lora(hh), vv + attn.v_memory_lora(hh)
    k = attn.norm_k(kk).view(b, m * s, n, d)
    v = vv.view(b, m * s, n, d)
    return _rope_apply(k, rope_hist).type_as(v), v


# ======== ours (distance gate) ========
# Pages of a gated model. The page of frame j in one layer is its four PARTS (W_k x, W_v x, down_k(x), down_v(x))
# in the dtype the layer computes in (K is built from them before norm_k, so they are not rounded to flash-attn's
# dtype), and the reader builds K, V at its own distance (CausalWanSelfAttention._gated_history). The slot of
# page j is frame j: pages are appended frame by frame from frame 0, so the reader of a history of m pages is
# frame m, the page in slot j sits at RoPE position j, and d = m - j.
GATED_PAGE_PARTS = 4


class _PartsTable:
    """_PageTable of a gated layer: one contiguous buffer [B, max_frames * L, C] per part, page j in rows
    [j * L, (j + 1) * L), allocated on first use and only ever appended to. Also holds the prompt's
    cross-attention K/V of the forwards without gradient (context_kv), as _PageTable does."""

    def __init__(self, max_frames):
        self.max_frames = max_frames
        self.parts = None
        self.tokens = None
        self.context_kv = None
        # the history K, V the forwards WITHOUT gradient of the current frame read: every such forward of a frame
        # (its denoising steps, the write of a critic-step rollout) reads the same pages at the same distances, so
        # the first builds them and the others reuse them; keyed by (pages, tokens per frame), dropped by end_rollout
        self.hist_kv = None

    def put(self, slot, *parts):
        L = parts[0].shape[1]
        if self.parts is None:
            self.tokens = L
            self.parts = tuple(p.new_empty((p.shape[0], self.max_frames * L, *p.shape[2:])) for p in parts)
        for buf, p in zip(self.parts, parts):
            buf[:, slot * L:(slot + 1) * L].copy_(p)

    def view(self, length):
        return tuple(buf[:, :length * self.tokens] for buf in self.parts)


class GatedPageRead(NamedTuple):
    """PageRead of a gated layer: the table, the number of pages, and per page the handles of its four parts
    (empty when the pages carry no graph)."""
    table: _PartsTable
    length: int
    handles: tuple


class _PageParts(torch.autograd.Function):
    """One part of a gated layer's history pages, [B, length * L, C], for a forward with gradient: a fresh copy of
    the table's rows, whose backward hands each page's handle its slice of the gradient as its own contiguous
    tensor (the reason _PageCat gives). The copy also keeps every op that saves its input for the backward pass
    (the gate, norm_k) off the page table, which later pages are written into."""

    @staticmethod
    def forward(ctx, history, *handles):
        ctx.sizes = [h.shape[1] for h in handles]
        return history.clone(memory_format=torch.contiguous_format)

    @staticmethod
    def backward(ctx, grad):
        pages = grad.split(ctx.sizes, dim=1) if ctx.sizes else ()
        return (None, *(g.clone(memory_format=torch.contiguous_format) for g in pages))


class GatedPersistentKVStore(PersistentKVStore):
    """PersistentKVStore of a model with a Memory-LoRA distance gate: a page is the four parts of a memory write
    (GATED_PAGE_PARTS), kept in a _PartsTable per layer; with page_mode "h" the writer with gradient appends
    (parts detached, h) and the h page carries the graph, as in PersistentKVStore."""

    def __init__(self, num_layers, max_frames, page_mode="kv"):
        super().__init__(num_layers, max_frames, page_mode)
        self.tables = [_PartsTable(max_frames) for _ in range(num_layers)]
        self.handles = [[] for _ in range(num_layers)]

    def append_all_layers(self, flat_pages):
        """flat_pages: (ka0, va0, zk0, zv0, ka1, ...), or (ka0, va0, zk0, zv0, h0, ...) with the parts detached
        from a writer with gradient under page_mode "h"."""
        width = len(flat_pages) // self.num_layers
        assert width in (GATED_PAGE_PARTS, GATED_PAGE_PARTS + 1), width
        for layer_idx in range(self.num_layers):
            parts = flat_pages[width * layer_idx:width * layer_idx + GATED_PAGE_PARTS]
            with torch.no_grad():
                self.tables[layer_idx].put(self.length, *parts)
            if width == GATED_PAGE_PARTS + 1:
                self.h_pages[layer_idx].append(flat_pages[width * layer_idx + GATED_PAGE_PARTS])
            elif parts[0].requires_grad:
                self.handles[layer_idx].append(tuple(_PageHandle.apply(p) for p in parts))
        self.length += 1

    def snapshot_all_layers(self):
        if self.page_mode == "h" and torch.is_grad_enabled():
            return tuple(HPageRead(tuple(self.h_pages[l])) for l in range(self.num_layers))
        return tuple(GatedPageRead(self.tables[l], self.length, tuple(self.handles[l]))
                     for l in range(self.num_layers))

    def table_all_layers(self):
        return tuple(GatedPageRead(self.tables[l], self.length, ()) for l in range(self.num_layers))

    def end_rollout(self):
        super().end_rollout()
        for table in self.tables:
            table.hist_kv = None
# ==== ours end ====


def _rope_table(freqs, grid_sizes, start_frame, frames=1):
    """The multiplier causal_rope_apply builds on every call, [L, 1, c] complex128,
    for a frame grid at absolute frame `start_frame` (every sample has the same
    grid), or for `frames` such grids in a row. Built once per persistent
    forward and shared by all its layers."""
    f, h, w = grid_sizes[0].tolist()
    f = f * frames
    c = freqs.size(1)
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    return torch.cat([
        freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1)


def _rope_apply(x, rope):
    """causal_rope_apply for x [B, L, n, d] whose L tokens are the grid of `rope`
    (_rope_table): the same float64 complex product, without the table rebuilt
    per call and without the cat with an empty tail and the stack, which copy
    the float64 result twice. Bitwise the same output and input gradient."""
    b, s, n, _ = x.shape
    x_c = torch.view_as_complex(x.to(torch.float64).reshape(b, s, n, -1, 2))
    return torch.view_as_real(x_c * rope).flatten(3).type_as(x)


_CU_SEQLENS = {}


def _cu_seqlens(b, length, device):
    key = (b, length, device)
    if key not in _CU_SEQLENS:
        _CU_SEQLENS[key] = torch.arange(0, (b + 1) * length, length, dtype=torch.int32, device=device)
    return _CU_SEQLENS[key]


def _flash_attention(q, k, v):
    """attention(q, k, v) for unpadded sequences (q_lens = k_lens = None): the
    same flash-attn call on the same arguments. attention() builds the two
    cu_seqlens on every call (two host-to-device copies and, on the GPU, two
    fills, two cats and two cumsums); here they are built once per
    (batch, length)."""
    fa = _attention_backend
    if not (fa.FLASH_ATTN_2_AVAILABLE or fa.FLASH_ATTN_3_AVAILABLE):
        return attention(q, k, v)
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.bfloat16)

    q, k, v = half(q.flatten(0, 1)), half(k.flatten(0, 1)), half(v.flatten(0, 1))
    q, k = q.to(v.dtype), k.to(v.dtype)
    cu_q, cu_k = _cu_seqlens(b, lq, q.device), _cu_seqlens(b, lk, q.device)
    if fa.FLASH_ATTN_3_AVAILABLE:
        x = fa.flash_attn_interface.flash_attn_varlen_func(
            q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=lq, max_seqlen_k=lk,
            softmax_scale=None, causal=False, deterministic=False)[0].unflatten(0, (b, lq))
    else:
        x = fa.flash_attn.flash_attn_varlen_func(
            q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=lq, max_seqlen_k=lk,
            dropout_p=0., softmax_scale=None, causal=False, window_size=(-1, -1),
            deterministic=False).unflatten(0, (b, lq))
    return x.type(out_dtype)
# ==== ours end ====


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 # ======== ours ========
                 memory_lora=False,
                 memory_lora_distance_gate=None
                 # ==== ours end ====
                 ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # ── Streaming long-video KV cache (relative RoPE) ──────────────────
        # When `kv_rope_relative` is True the self-attention stores *un-roped*
        # K/V in the cache and re-applies RoPE every step with contiguous
        # relative frame indices, so the temporal position stays in
        # [0, kv_cache_max_frames-1] (never exceeding the trained range).
        # `kv_cache_max_frames` is the size of the attention window *including*
        # the frame currently being generated: with the defaults below the
        # window is the sink frame (idx 0) + 19 FIFO context frames (idx 1-19)
        # + the current frame (idx 20). The first `kv_cache_sink` frame(s) are
        # never evicted; the rest roll FIFO.
        # Default off → ordinary (short-video) inference is unchanged.
        self.kv_rope_relative = False
        self.kv_cache_max_frames = 21
        self.kv_cache_sink = 1
        # Trained temporal range (position-encoding ceiling). RoPE positions are
        # placed to match training: sink at {0..sink-1}, recent window pinned to
        # the top of this range, gap preserved; beyond it the geometry freezes.
        self.kv_cache_train_frames = 21
        self.kv_cache_position_mode = "top_aligned"

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        # ======== ours ========
        # Memory-LoRA: the memory writer's K/V projection is (W + dW), applied
        # to W_k h / W_v h BEFORE norm_k, i.e. really (W + dW) h.
        self.memory_lora = memory_lora
        # distance gate (see MemoryLoRA.gated); None = the ungated Memory-LoRA, code paths unchanged
        self.distance_gate = memory_lora_distance_gate_mode(memory_lora_distance_gate)
        if self.distance_gate is not None and not memory_lora:
            raise ValueError("memory_lora_distance_gate needs model_kwargs.memory_lora: true")
        if memory_lora:
            self.k_memory_lora = MemoryLoRA(dim, distance_gate=self.distance_gate)
            self.v_memory_lora = MemoryLoRA(dim, distance_gate=self.distance_gate)
        # ==== ours end ====

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        # ======== ours ========
        persistent=None,
        memory_write=False
        # ==== ours end ====
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        # ======== ours ========
        if persistent is not None:
            return self.forward_persistent(x, **persistent)
        # ==== ours end ====
        # ======== ours (distance gate) ========
        if self.distance_gate is not None and kv_cache is not None:
            return self._forward_cache_gated(x, grid_sizes, freqs, kv_cache, current_start, memory_write)
        # ==== ours end ====
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # ======== ours ========
        # Memory-LoRA routing. Teacher forcing needs no flag: the clean half IS
        # the memory and the noisy half IS the generation. The cached path has
        # no such split and marks its context refresh with memory_write.
        dk = dv = None
        if self.memory_lora:
            is_tf_seq = kv_cache is None and (s == seq_lens[0].item() * 2)
            if is_tf_seq and self.distance_gate is not None:      # (distance gate)
                raise NotImplementedError("memory_lora_distance_gate: the teacher-forced forward attends to every "
                                          "clean frame at once and cannot give a key a K per reader")
            if is_tf_seq or (kv_cache is not None and memory_write):
                dk, dv = self.k_memory_lora(x), self.v_memory_lora(x)
                if is_tf_seq:
                    keep = torch.zeros(1, s, 1, device=x.device, dtype=dk.dtype)
                    keep[:, :s // 2] = 1.0        # clean half only
                    dk, dv = dk * keep, dv * keep
        # ==== ours end ====

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            # ======== ours ========
            # (replaces upstream's `k = norm_k(k(x))`, `v = v(x)`; identical when dk is None)
            kk, vv = self.k(x), self.v(x)
            if dk is not None:
                kk, vv = kk + dk, vv + dv
            k = self.norm_k(kk).view(b, s, n, d)
            v = vv.view(b, s, n, d)
            # ==== ours end ====
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        elif self.kv_rope_relative:
            # ── Streaming long-video path ──────────────────────────────────
            # Store raw (pre-RoPE) K/V; re-apply RoPE with relative indices so
            # the temporal position always lives in [0, kv_cache_max_frames-1].
            fs = math.prod(grid_sizes[0][1:]).item()
            cur_frames = q.shape[1] // fs
            cur_abs = current_start // fs
            max_frames = self.kv_cache_max_frames
            sink = self.kv_cache_sink

            # Per-cache bookkeeping: chronological list of the absolute frame
            # index occupying each slot (sink frames first, then recent window).
            abs_list = kv_cache.get("stream_abs", None)
            if abs_list is None:
                abs_list = []
                kv_cache["stream_abs"] = abs_list

            new_block_abs = [cur_abs + i for i in range(cur_frames)]
            # Re-running the same frame (extra denoising / context-update step)
            # overwrites the tail slot instead of appending a new frame.
            is_rewrite = (len(abs_list) >= cur_frames
                          and abs_list[-cur_frames:] == new_block_abs)

            if is_rewrite:
                tail_slot = len(abs_list) - cur_frames
            else:
                n_after = len(abs_list) + cur_frames
                if n_after > max_frames:
                    # Evict the oldest non-sink frame(s); shift the kept window
                    # left, leaving the first `sink` frame(s) pinned in place.
                    num_evict = n_after - max_frames
                    src_lo = (sink + num_evict) * fs
                    src_hi = len(abs_list) * fs
                    dst_lo = sink * fs
                    dst_hi = dst_lo + (src_hi - src_lo)
                    if src_hi > src_lo:
                        kv_cache["k"][:, dst_lo:dst_hi] = kv_cache["k"][:, src_lo:src_hi].clone()
                        kv_cache["v"][:, dst_lo:dst_hi] = kv_cache["v"][:, src_lo:src_hi].clone()
                    del abs_list[sink:sink + num_evict]
                abs_list.extend(new_block_abs)
                tail_slot = len(abs_list) - cur_frames

            # Write the current block's *un-roped* K/V into its tail slot.
            tail_lo = tail_slot * fs
            tail_hi = tail_lo + cur_frames * fs
            kv_cache["k"][:, tail_lo:tail_hi] = k
            kv_cache["v"][:, tail_lo:tail_hi] = v

            num_occ = len(abs_list)
            if self.kv_cache_position_mode == "contiguous":
                # Compress the retained cache into the training prefix. With
                # sink=0, max_frames=12 this yields {0..11} once full and then
                # keeps that exact RoPE geometry forever.
                pos = torch.arange(num_occ, device=q.device, dtype=torch.long)
            else:
                # RoPE positions that MATCH the teacher-forcing training geometry:
                #   sink frames pinned at {0 .. sink-1};
                #   recent window pinned to the TOP of the trained range so the
                #     query/window relative distances equal those seen in training;
                #   the gap (dropped middle frames) is preserved.
                # Query is clamped to train_frames-1, so beyond the trained range
                # the whole geometry freezes (e.g. sink={0,1,2}, window={9..20}
                # forever) instead of being compressed into {0..num_occ-1}.
                n_win = num_occ - sink
                if n_win <= 0:
                    pos = torch.arange(num_occ, device=q.device, dtype=torch.long)
                else:
                    q_abs = cur_abs + cur_frames - 1
                    Q = min(int(q_abs), self.kv_cache_train_frames - 1)
                    pos = torch.cat([
                        torch.arange(sink, device=q.device, dtype=torch.long),
                        torch.arange(Q - (n_win - 1), Q + 1, device=q.device, dtype=torch.long),
                    ])

            window_grid = grid_sizes.clone()
            window_grid[:, 0] = num_occ
            roped_key = causal_rope_apply_frames(
                kv_cache["k"][:, :num_occ * fs], window_grid, freqs, pos).type_as(v)

            q_pos = pos[tail_slot:tail_slot + cur_frames]
            roped_query = causal_rope_apply_frames(
                q, grid_sizes, freqs, q_pos).type_as(v)

            x = attention(
                roped_query,
                roped_key,
                kv_cache["v"][:, :num_occ * fs]
            )
            # Keep the legacy counters roughly in sync (unused by this path).
            kv_cache["global_end_index"].fill_(current_start + q.shape[1])
            kv_cache["local_end_index"].fill_(num_occ * fs)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            win_lo = max(0, local_end_index - self.max_attention_size)
            if self.local_attn_size != -1 and sink_tokens > 0 and win_lo > sink_tokens:
                # Disjoint sink + recent window (a real gap separates them).
                # Eviction above keeps [0, sink_tokens) un-rolled, so this slice is
                # always the original first `sink_size` frames.
                k_read = torch.cat(
                    [kv_cache["k"][:, :sink_tokens], kv_cache["k"][:, win_lo:local_end_index]], dim=1)
                v_read = torch.cat(
                    [kv_cache["v"][:, :sink_tokens], kv_cache["v"][:, win_lo:local_end_index]], dim=1)
            else:
                # Contiguous: when a sink exists and the window already reaches back
                # into/before it, read from 0 so the sink is not skipped (e.g. frame
                # 12 must see frame 0). Otherwise the plain recent slice (pure window,
                # or full attention when local_attn_size == -1).
                lo = 0 if (self.local_attn_size != -1 and sink_tokens > 0) else win_lo
                k_read = kv_cache["k"][:, lo:local_end_index]
                v_read = kv_cache["v"][:, lo:local_end_index]
            x = attention(roped_query, k_read, v_read)
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        # x.shape is [1, 65520, 1536]
        x = self.o(x)
        return x

    # ======== ours (distance gate) ========
    def _gated_kv(self, ka, va, zk, zv, dist, mem=None):
        """K (before RoPE) and V [B, m * L, n, d] of m frames of L tokens from their parts ka = W_k x, va = W_v x
        [B, m * L, dim] and zk = down_k(x), zv = down_v(x) [B, m * L, rank], frame j read at distance dist[j]:
        K = norm_k(W_k x + dW_k(d) x), V = W_v x + dW_v(d) x. mem [m] (0 or 1) drops the Memory-LoRA of a frame
        that was not a memory write (inference cache only)."""
        b, L = ka.shape[:2]
        m = dist.numel()
        s = L // m
        dk = self.k_memory_lora.gated(zk.reshape(b, m, s, -1), dist)
        dv = self.v_memory_lora.gated(zv.reshape(b, m, s, -1), dist)
        if mem is not None:
            mem = mem.to(dk.dtype).view(1, m, 1, 1)
            dk, dv = dk * mem, dv * mem
        kk = ka + dk.view(b, L, -1)
        vv = va + dv.view(b, L, -1)
        return self.norm_k(kk).view(b, L, self.num_heads, self.head_dim), vv.view(b, L, self.num_heads, self.head_dim)

    def _mix_kv(self, kL, k0, v0, zv, dist, mem=None):
        """mix: K, V [B, m * L, n, d] of m frames of L tokens read at distances dist [m]:
        K = g_d K^L + (1 - g_d) K^0 with K^L, K^0 after norm_k and RoPE [B, m * L, n, d], computed in their dtype;
        V = V^0 + g'_d (up_v(z_v) * scale), V^0 = W_v x [B, m * L, n, d], z_v = down_v(x) [B, m * L, rank];
        g = sigmoid(theta) of the K / V gate. g = 1 gives K^L and W_v x + dW_v x as the ungated memory write computes
        them (a factor 1 and a term 0 aside, op for op), g = 0 gives K^0, W_v x. mem [m] (0/1, inference cache only)
        multiplies g: a slot that was not a memory write reads as K^0, V^0."""
        b, Lt = kL.shape[:2]
        m = dist.numel()
        s = Lt // m
        n, d = self.num_heads, self.head_dim
        gk, gv = self.k_memory_lora.mix_weight(dist), self.v_memory_lora.mix_weight(dist)
        if mem is not None:
            gk, gv = gk * mem.to(gk.dtype), gv * mem.to(gv.dtype)
        gk = gk.to(kL.dtype).view(1, m, 1, 1, 1)
        k = gk * kL.reshape(b, m, s, n, d) + (1 - gk) * k0.reshape(b, m, s, n, d)
        u = self.v_memory_lora.up(zv.reshape(b, m, s, -1)) * self.v_memory_lora.scale
        v = v0.reshape(b, m, s, n * d) + u * gv.to(u.dtype).view(1, m, 1, 1)
        return k.view(b, Lt, n, d), v.view(b, Lt, n, d)

    def _mix_parts_from_h(self, hh, rope_hist):
        """mix: the page parts (K^L, K^0 roped and in the page dtype, V^0, z_v) of h pages hh [B, m * L, dim], with the
        writers' ops (h pages with gradient)."""
        b, Lt = hh.shape[:2]
        n, d = self.num_heads, self.head_dim
        kk, vv = self.k(hh), self.v(hh)
        kkL = kk + self.k_memory_lora(hh)
        v0 = vv.view(b, Lt, n, d)
        ad = _attn_dtype(v0.dtype)
        kL = _rope_apply(self.norm_k(kkL).view(b, Lt, n, d), rope_hist).type_as(v0).to(ad)
        k0 = _rope_apply(self.norm_k(kk).view(b, Lt, n, d), rope_hist).type_as(v0).to(ad)
        return kL, k0, v0, self.v_memory_lora.down(hh)

    @staticmethod
    def _gated_own(lora, z):
        """dW(0) x [B, L, dim] of a frame's own K or V (d = 0) from z = down(x) [B, L, rank]: the GEMMs of
        lora(x) on the same rows, so with the identity gate it is lora(x) bit for bit."""
        b, s = z.shape[:2]
        d0 = torch.zeros(1, dtype=torch.long, device=z.device)
        return lora.gated(z.view(b, 1, s, -1), d0).view(b, s, -1)

    def _forward_cache_gated(self, x, grid_sizes, freqs, kv_cache, current_start, memory_write):
        """forward() with a KV cache for a model with a Memory-LoRA distance gate (inference). The same cache
        bookkeeping as the plain path (sink + local window, eviction) and as the --long_video path (kv_rope_relative:
        positions relative to the window), on these per-token buffers of the cache dict:
          k, v     W_k x, W_v x before norm_k and RoPE (the dict's own "k", "v" buffers)
          zk, zv   down_k(x), down_v(x)                 (allocated here on first use)
          mem      1 where the slot was written by a memory write (memory_write=True), else 0
          pos      the RoPE frame position the slot was written at (plain path)
        Every forward rebuilds the K, V of the frames it reads for ITS query position q: frame j at RoPE position
        p_j is read at d = clamp(q - p_j, 0, 20), K_j = RoPE_{p_j}(norm_k(W_k x_j + mem_j dW_k(d) x_j)),
        V_j = W_v x_j + mem_j dW_v(d) x_j. The current frame is one of them (d = 0, mem = memory_write), so the
        generation forwards attend to their own K/V without the Memory-LoRA, as before. With the identity gate
        this is the ungated path to rounding (the up projection runs over all read frames at once)."""
        if torch.is_grad_enabled():
            raise RuntimeError("memory_lora_distance_gate: the KV-cache path is for inference (no gradient); "
                               "train with grad_mode kv_forcing")
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        fs = math.prod(grid_sizes[0][1:]).item()
        if s != fs:
            raise NotImplementedError("memory_lora_distance_gate: one frame per forward (num_frame_per_block 1)")
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        ka, va = self.k(x), self.v(x)
        cur = current_start // fs
        mix = self.distance_gate == "mix"
        zv = self.v_memory_lora.down(x)
        if mix:
            # mix: K^L, K^0 after norm_k ("k", "k0"; after RoPE at the slot's position in the plain path, which keeps
            # roped K as ungated; the --long_video path ropes per read, as ungated), V^0 ("v"), z_v. A forward that is
            # not a memory write keeps K^0 in both (it is read with g = 0).
            k0 = self.norm_k(ka).view(b, s, n, d)
            kL = self.norm_k(ka + self.k_memory_lora(x)).view(b, s, n, d) if memory_write else k0
            if not self.kv_rope_relative:
                k0 = causal_rope_apply(k0, grid_sizes, freqs, start_frame=cur).type_as(va)
                kL = causal_rope_apply(kL, grid_sizes, freqs, start_frame=cur).type_as(va) if memory_write else k0
            new = dict(k=kL, v=va.view(b, s, n, d), k0=k0, zv=zv)
            extras = {"k0": (n, d), "zv": (zv.shape[-1],)}
        else:
            zk = self.k_memory_lora.down(x)
            new = dict(k=ka.view(b, s, n, d), v=va.view(b, s, n, d), zk=zk, zv=zv)
            extras = {"zk": (zk.shape[-1],), "zv": (zv.shape[-1],)}
        buf_k = kv_cache["k"]
        size = buf_k.shape[1]
        for key, tail in extras.items():
            if kv_cache.get(key) is None or tuple(kv_cache[key].shape) != (buf_k.shape[0], size, *tail):
                kv_cache[key] = buf_k.new_zeros(buf_k.shape[0], size, *tail)
        if kv_cache.get("mem") is None or kv_cache["mem"].shape[1] != size:
            kv_cache["mem"] = buf_k.new_zeros(1, size)
            kv_cache["pos"] = torch.zeros(1, size, dtype=torch.long, device=buf_k.device)
        names = ("k", "v") + tuple(extras) + ("mem", "pos")

        def write(lo, hi):
            for key, val in new.items():
                kv_cache[key][:, lo:hi] = val
            kv_cache["mem"][:, lo:hi] = 1.0 if memory_write else 0.0
            kv_cache["pos"][:, lo:hi] = cur

        def shift(dst_lo, src_lo, length):
            for key in names:
                kv_cache[key][:, dst_lo:dst_lo + length] = kv_cache[key][:, src_lo:src_lo + length].clone()

        if self.kv_rope_relative:
            # the --long_video bookkeeping of forward(), on all buffers
            max_frames, sink = self.kv_cache_max_frames, self.kv_cache_sink
            abs_list = kv_cache.get("stream_abs", None)
            if abs_list is None:
                abs_list = []
                kv_cache["stream_abs"] = abs_list
            new_block_abs = [cur]
            if len(abs_list) >= 1 and abs_list[-1:] == new_block_abs:
                tail_slot = len(abs_list) - 1
            else:
                n_after = len(abs_list) + 1
                if n_after > max_frames:
                    num_evict = n_after - max_frames
                    src_lo, src_hi, dst_lo = (sink + num_evict) * fs, len(abs_list) * fs, sink * fs
                    if src_hi > src_lo:
                        shift(dst_lo, src_lo, src_hi - src_lo)
                    del abs_list[sink:sink + num_evict]
                abs_list.extend(new_block_abs)
                tail_slot = len(abs_list) - 1
            write(tail_slot * fs, (tail_slot + 1) * fs)
            num_occ = len(abs_list)
            if self.kv_cache_position_mode == "contiguous":
                pos = torch.arange(num_occ, device=q.device, dtype=torch.long)
            else:
                n_win = num_occ - sink
                if n_win <= 0:
                    pos = torch.arange(num_occ, device=q.device, dtype=torch.long)
                else:
                    Q = min(int(cur), self.kv_cache_train_frames - 1)
                    pos = torch.cat([
                        torch.arange(sink, device=q.device, dtype=torch.long),
                        torch.arange(Q - (n_win - 1), Q + 1, device=q.device, dtype=torch.long),
                    ])
            read = {key: kv_cache[key][:, :num_occ * fs] for key in names[:5]}
            q_pos = pos[tail_slot:tail_slot + 1]
            roped_query = causal_rope_apply_frames(q, grid_sizes, freqs, q_pos).type_as(va)
            pos_frames = pos
            new_global, new_local = current_start + s, num_occ * fs
        else:
            # the plain bookkeeping of forward(), on all buffers
            current_end = current_start + s
            sink_tokens = self.sink_size * fs
            local_end, global_end = kv_cache["local_end_index"].item(), kv_cache["global_end_index"].item()
            if self.local_attn_size != -1 and current_end > global_end and s + local_end > size:
                num_evicted = s + local_end - size
                num_rolled = local_end - num_evicted - sink_tokens
                shift(sink_tokens, sink_tokens + num_evicted, num_rolled)
                local_end_index = local_end + current_end - global_end - num_evicted
            else:
                local_end_index = local_end + current_end - global_end
            write(local_end_index - s, local_end_index)
            win_lo = max(0, local_end_index - self.max_attention_size)
            if self.local_attn_size != -1 and sink_tokens > 0 and win_lo > sink_tokens:
                read = {key: torch.cat([kv_cache[key][:, :sink_tokens], kv_cache[key][:, win_lo:local_end_index]],
                                       dim=1) for key in names}
            else:
                lo = 0 if (self.local_attn_size != -1 and sink_tokens > 0) else win_lo
                read = {key: kv_cache[key][:, lo:local_end_index] for key in names}
            q_pos = torch.full((1,), cur, dtype=torch.long, device=q.device)
            roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=cur).type_as(va)
            pos_frames = read["pos"][0, ::fs]
            new_global, new_local = current_end, local_end_index
        L = read["k"].shape[1]
        if L % fs:
            raise RuntimeError("memory_lora_distance_gate: the cache read is not whole frames")
        m = L // fs
        dist = (q_pos - pos_frames).clamp(0, MEMORY_LORA_GATE_DISTANCES - 1)
        window_grid = grid_sizes.clone()
        window_grid[:, 0] = m
        if mix:
            kL_r, k0_r = read["k"].to(va.dtype), read["k0"].to(va.dtype)
            if self.kv_rope_relative:
                kL_r = causal_rope_apply_frames(kL_r, window_grid, freqs, pos_frames).type_as(va)
                k0_r = causal_rope_apply_frames(k0_r, window_grid, freqs, pos_frames).type_as(va)
            roped_key, v_r = self._mix_kv(kL_r, k0_r, read["v"].to(va.dtype), read["zv"].to(zv.dtype), dist,
                                          mem=read["mem"][0, ::fs])
        else:
            k_r, v_r = self._gated_kv(read["k"].reshape(b, L, n * d).to(ka.dtype), read["v"].reshape(b, L, n * d).to(va.dtype),
                                      read["zk"].to(zv.dtype), read["zv"].to(zv.dtype), dist, mem=read["mem"][0, ::fs])
            roped_key = causal_rope_apply_frames(k_r, window_grid, freqs, pos_frames).type_as(v_r)
        x = attention(roped_query, roped_key, v_r)
        kv_cache["global_end_index"].fill_(new_global)
        kv_cache["local_end_index"].fill_(new_local)
        return self.o(x.flatten(2))
    # ==== ours end ====

    # ======== ours ========
    def forward_persistent(self, x, rope, history, write_page=False, kv_only=False,
                           write_mask=None, rope_hist=None):
        """Attention over immutable history pages plus this frame's own K/V.

        history is the layer's PageRead. With gradient, the history slots and
        this frame's K/V are concatenated here rather than by the caller, so a
        checkpointed frame saves references instead of a copy of the whole
        history. Without gradient nothing is concatenated while the table has a
        free slot: this frame's K/V go into it and the attention runs on a view
        of the table (see PersistentKVStore).
        Returns y, plus this frame's (k, v) when write_page is set; with kv_only
        just (k, v), skipping the attention. Pages hold post-RoPE K, so a page
        is interchangeable with a KV-cache slot. `rope` is the frame's
        _rope_table.

        Under page_mode "h" a forward with gradient is handed an HPageRead: the
        history K/V are rebuilt from the h pages (`rope_hist` is the
        _rope_table of the history frames), and the page it writes is
        (k, v, h) with k, v detached for the page table and h = x, this
        layer's attention input, carrying the graph.
        A write_mask marks the fused [noisy; write] batch
        (forward_persistent_pair).
        """
        if write_mask is not None:
            return self.forward_persistent_pair(x, rope, history, write_mask, rope_hist)
        # ======== ours (distance gate) ========
        if self.distance_gate is not None:
            return self._forward_persistent_gated(x, rope, history, write_page, kv_only, rope_hist)
        # ==== ours end ====
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if not kv_only:
            q = self.norm_q(self.q(x)).view(b, s, n, d)
        kk, vv = self.k(x), self.v(x)
        if self.memory_lora and write_page:      # the page write IS the memory write
            kk, vv = kk + self.k_memory_lora(x), vv + self.v_memory_lora(x)
        k = self.norm_k(kk).view(b, s, n, d)
        v = vv.view(b, s, n, d)
        if not kv_only:
            q_rope = _rope_apply(q, rope).type_as(v)
        k_rope = _rope_apply(k, rope).type_as(v)
        if isinstance(history, HPageRead):
            if not torch.is_grad_enabled():      # the table is what a forward without gradient reads
                raise RuntimeError("h pages are for forwards with gradient.")
            page = (k_rope.detach(), v.detach(), x)
            if kv_only:
                return page
            if len(history.pages):
                k_hist, v_hist = _rebuild_history_kv(self, history.pages, rope_hist)
                k_attn, v_attn = torch.cat([k_hist, k_rope], dim=1), torch.cat([v_hist, v], dim=1)
            else:
                k_attn, v_attn = k_rope, v
            y = self.o(_flash_attention(q_rope, k_attn, v_attn).flatten(2))
            return (y, *page) if write_page else y
        table, length = history.table, history.length
        if not torch.is_grad_enabled() and length < table.max_frames:     # the history is read in place
            k_rope, v = table.put(length, k_rope, v)
            if kv_only:
                return k_rope, v
            k_attn, v_attn = table.k[:, :(length + 1) * s], table.v[:, :(length + 1) * s]
        elif kv_only:
            return k_rope, v
        elif length:
            k_attn = _PageCat.apply(table.k[:, :length * s], k_rope, *history.k_handles)
            v_attn = _PageCat.apply(table.v[:, :length * s], v, *history.v_handles)
        else:
            k_attn, v_attn = k_rope, v
        y = self.o(_flash_attention(q_rope, k_attn, v_attn).flatten(2))
        if write_page:
            return y, k_rope, v
        return y

    def forward_persistent_pair(self, x, rope, history, write_mask, rope_hist=None):
        """forward_persistent of the batch [noisy; write] in one call, with
        gradient: the first half is the noisy forward, the second half the page
        write. Both attend to the same history plus their OWN K/V, never to each
        other's. write_mask (0 for the noisy half, 1 for the write half)
        confines the Memory-LoRA to the write; adding 0 leaves the noisy half
        exact. Returns y and the write half's page. RoPE runs per half, so the K
        page is a fresh tensor; the V page (and the h page) is copied out of the
        fused batch, which a view would pin.
        """
        # ======== ours (distance gate) ========
        if self.distance_gate is not None:
            return self._forward_persistent_pair_gated(x, rope, history, write_mask, rope_hist)
        # ==== ours end ====
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        h = b // 2
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        kk, vv = self.k(x), self.v(x)
        if self.memory_lora:
            kk, vv = kk + self.k_memory_lora(x) * write_mask, vv + self.v_memory_lora(x) * write_mask
        k = self.norm_k(kk).view(b, s, n, d)
        v = vv.view(b, s, n, d)
        q_rope = _rope_apply(q, rope).type_as(v)
        k_noisy = _rope_apply(k[:h], rope).type_as(v)
        k_write = _rope_apply(k[h:], rope).type_as(v)
        v_noisy, v_write = v[:h], v[h:].clone()
        if isinstance(history, HPageRead):
            page = (k_write.detach(), v_write.detach(), x[h:].clone())
            if len(history.pages):
                k_hist, v_hist = _rebuild_history_kv(self, history.pages, rope_hist)
                k_attn = torch.cat([torch.cat([k_hist, k_noisy], dim=1), torch.cat([k_hist, k_write], dim=1)])
                v_attn = torch.cat([torch.cat([v_hist, v_noisy], dim=1), torch.cat([v_hist, v_write], dim=1)])
            else:
                k_attn, v_attn = torch.cat([k_noisy, k_write]), torch.cat([v_noisy, v_write])
        else:
            page = (k_write, v_write)
            table, length = history.table, history.length
            if length:
                k_attn = _PageCatPair.apply(table.k[:, :length * s], k_noisy, k_write, *history.k_handles)
                v_attn = _PageCatPair.apply(table.v[:, :length * s], v_noisy, v_write, *history.v_handles)
            else:
                k_attn, v_attn = torch.cat([k_noisy, k_write]), torch.cat([v_noisy, v_write])
        y = self.o(_flash_attention(q_rope, k_attn, v_attn).flatten(2))
        return (y, *page)

    # ======== ours (distance gate) ========
    def _gated_history(self, history, s, rope_hist):
        """The history K (after RoPE) and V [B, m * s, n, d] of a gated layer as the frame being computed reads
        them, or None for an empty history. A history of m pages is read by frame m (the pages are frames
        0..m-1, slot j = RoPE position j, `rope_hist` the RoPE table of frames 0..m-1), so page j is at
        d = min(m - j, 20). The parts come from the page table (kv pages; with gradient through _PageParts,
        which routes each page's gradient to its handles) or are recomputed from the h pages with the writers'
        ops; either way K, V are built inside the reader's forward and checkpoint region."""
        mix = self.distance_gate == "mix"
        if isinstance(history, HPageRead):
            m = len(history.pages)
            if not m:
                return None
            hh = _HPageCat.apply(*history.pages)
            if mix:
                parts = self._mix_parts_from_h(hh, rope_hist)
            else:
                parts = (self.k(hh), self.v(hh), self.k_memory_lora.down(hh), self.v_memory_lora.down(hh))
        else:
            if not isinstance(history, GatedPageRead):
                raise TypeError("a gated layer reads GatedPersistentKVStore pages; got %s" % type(history).__name__)
            m = history.length
            if not m:
                return None
            if not torch.is_grad_enabled() and not mix:
                cached = history.table.hist_kv
                if cached is not None and cached[0] == (m, s):
                    return cached[1]
            views = history.table.view(m)
            if torch.is_grad_enabled():
                if len(history.handles) not in (0, m):
                    raise RuntimeError("distance gate: %d page handles for %d pages" % (len(history.handles), m))
                parts = tuple(_PageParts.apply(view, *(hd[i] for hd in history.handles))
                              for i, view in enumerate(views))
            else:
                parts = views
        if rope_hist is None or rope_hist.shape[0] != m * s:
            raise RuntimeError("distance gate: the history must be the pages of frames 0..m-1, read by frame m")
        dist = torch.arange(m, 0, -1, device=parts[0].device).clamp_(max=MEMORY_LORA_GATE_DISTANCES - 1)
        if mix:                                  # pages written once, read with an elementwise mix
            return self._mix_kv(*parts, dist)
        k, v = self._gated_kv(*parts, dist)
        out = (_rope_apply(k, rope_hist).type_as(v), v)
        if not torch.is_grad_enabled() and isinstance(history, GatedPageRead):
            history.table.hist_kv = ((m, s), out)
        return out

    def _gated_history_checkpointed(self, history, s, rope_hist):
        """_gated_history, with gradient as its own (nested) activation-checkpoint region: what it builds (the parts'
        copies, W x + dW(d) x before norm_k, norm_k's float32 intermediates, the gated LoRA terms) is then recomputed
        in the backward pass instead of being kept for the whole recomputation of the enclosing block group; only the
        K, V it returns (which flash-attn keeps anyway) remain. The same operations, so the same values."""
        empty = (len(history.pages) == 0) if isinstance(history, HPageRead) else (history.length == 0)
        if empty or not torch.is_grad_enabled():
            return self._gated_history(history, s, rope_hist)
        return torch.utils.checkpoint.checkpoint(self._gated_history, history, s, rope_hist,
                                                 use_reentrant=False, preserve_rng_state=False)

    def _forward_persistent_gated(self, x, rope, history, write_page, kv_only, rope_hist):
        """forward_persistent of a gated layer. A page is the parts (W_k x, W_v x, down_k(x), down_v(x)), or, for
        h pages with gradient, (the parts detached, x). The frame's own K/V are W x (generation) or
        (W + dW(0)) x (memory write), as in the ungated layer; the history is _gated_history."""
        if self.distance_gate == "mix":
            return self._forward_persistent_mix(x, rope, history, write_page, kv_only, rope_hist)
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        h_pages = isinstance(history, HPageRead)
        if h_pages and not torch.is_grad_enabled():
            raise RuntimeError("h pages are for forwards with gradient.")
        kk, vv = self.k(x), self.v(x)
        page = ()
        if write_page:                           # the page write IS the memory write
            zk, zv = self.k_memory_lora.down(x), self.v_memory_lora.down(x)
            page = (kk, vv, zk, zv)
            if h_pages:
                page = tuple(p.detach() for p in page) + (x,)
            if kv_only:
                return page
            kk = kk + self._gated_own(self.k_memory_lora, zk)
            vv = vv + self._gated_own(self.v_memory_lora, zv)
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(kk).view(b, s, n, d)
        v = vv.view(b, s, n, d)
        q_rope = _rope_apply(q, rope).type_as(v)
        k_rope = _rope_apply(k, rope).type_as(v)
        hist = self._gated_history_checkpointed(history, s, rope_hist)
        if hist is None:
            k_attn, v_attn = k_rope, v
        else:
            k_attn, v_attn = torch.cat([hist[0], k_rope], dim=1), torch.cat([hist[1], v], dim=1)
        y = self.o(_flash_attention(q_rope, k_attn, v_attn).flatten(2))
        return (y, *page) if write_page else y

    def _forward_persistent_pair_gated(self, x, rope, history, write_mask, rope_hist):
        """forward_persistent_pair of a gated layer: the Memory-LoRA at d = 0 on the write half only (times
        write_mask, as ungated), one history K/V (both halves are the same frame, so they read it at the same
        distances), and the write half's parts as its page, copied out of the fused batch."""
        if self.distance_gate == "mix":
            return self._forward_persistent_pair_mix(x, rope, history, write_mask, rope_hist)
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        h = b // 2
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        ka, va = self.k(x), self.v(x)
        zk, zv = self.k_memory_lora.down(x), self.v_memory_lora.down(x)
        kk = ka + self._gated_own(self.k_memory_lora, zk) * write_mask
        vv = va + self._gated_own(self.v_memory_lora, zv) * write_mask
        k = self.norm_k(kk).view(b, s, n, d)
        v = vv.view(b, s, n, d)
        q_rope = _rope_apply(q, rope).type_as(v)
        k_noisy = _rope_apply(k[:h], rope).type_as(v)
        k_write = _rope_apply(k[h:], rope).type_as(v)
        v_noisy, v_write = v[:h], v[h:]
        if isinstance(history, HPageRead):
            page = tuple(p[h:].detach() for p in (ka, va, zk, zv)) + (x[h:].clone(),)
        else:
            page = tuple(p[h:].clone() for p in (ka, va, zk, zv))
        hist = self._gated_history_checkpointed(history, s, rope_hist)
        if hist is None:
            k_attn, v_attn = torch.cat([k_noisy, k_write]), torch.cat([v_noisy, v_write])
        else:
            k_hist, v_hist = hist
            k_attn = torch.cat([torch.cat([k_hist, k_noisy], dim=1), torch.cat([k_hist, k_write], dim=1)])
            v_attn = torch.cat([torch.cat([v_hist, v_noisy], dim=1), torch.cat([v_hist, v_write], dim=1)])
        y = self.o(_flash_attention(q_rope, k_attn, v_attn).flatten(2))
        return (y, *page)

    def _forward_persistent_mix(self, x, rope, history, write_page, kv_only, rope_hist):
        """forward_persistent of a mix layer. The memory write computes K^L (the ungated write's K, op for op), K^0, V^0
        and z_v and keeps them as its page (K^L, K^0 in the page dtype); it attends to itself at d = 0 with the mix of
        g_0. A generation forward computes and attends to W x, as ungated."""
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        h_pages = isinstance(history, HPageRead)
        if h_pages and not torch.is_grad_enabled():
            raise RuntimeError("h pages are for forwards with gradient.")
        kk, vv = self.k(x), self.v(x)
        v0 = vv.view(b, s, n, d)
        page = ()
        if write_page:
            zv = self.v_memory_lora.down(x)
            kL = _rope_apply(self.norm_k(kk + self.k_memory_lora(x)).view(b, s, n, d), rope).type_as(v0)
            k0 = _rope_apply(self.norm_k(kk).view(b, s, n, d), rope).type_as(v0)
            ad = _attn_dtype(v0.dtype)
            page = (kL.to(ad), k0.to(ad), v0, zv)
            if h_pages:
                page = tuple(p.detach() for p in page) + (x,)
            if kv_only:
                return page
            k_own, v_own = self._mix_kv(kL, k0, v0, zv, torch.zeros(1, dtype=torch.long, device=x.device))
        else:
            k_own, v_own = _rope_apply(self.norm_k(kk).view(b, s, n, d), rope).type_as(v0), v0
        q_rope = _rope_apply(self.norm_q(self.q(x)).view(b, s, n, d), rope).type_as(v0)
        hist = self._gated_history_checkpointed(history, s, rope_hist)
        if hist is None:
            k_attn, v_attn = k_own, v_own
        else:
            k_attn, v_attn = torch.cat([hist[0], k_own.to(hist[0].dtype)], dim=1), torch.cat([hist[1], v_own], dim=1)
        y = self.o(_flash_attention(q_rope, k_attn, v_attn).flatten(2))
        return (y, *page) if write_page else y

    def _forward_persistent_pair_mix(self, x, rope, history, write_mask, rope_hist):
        """forward_persistent_pair of a mix layer: the noisy half is a generation forward (W x; the Memory-LoRA term
        times write_mask = 0, as ungated), the write half a memory write (K^L, K^0, V^0, z_v; own K/V mixed at d = 0)."""
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        h = b // 2
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        kk, vv = self.k(x), self.v(x)
        zv = self.v_memory_lora.down(x)
        kL = self.norm_k(kk + self.k_memory_lora(x) * write_mask).view(b, s, n, d)
        k0 = self.norm_k(kk[h:]).view(h, s, n, d)
        v0 = vv.view(b, s, n, d)
        q_rope = _rope_apply(q, rope).type_as(v0)
        k_noisy = _rope_apply(kL[:h], rope).type_as(v0)
        kL_w = _rope_apply(kL[h:], rope).type_as(v0)
        k0_w = _rope_apply(k0, rope).type_as(v0)
        v_noisy, v0_w, zv_w = v0[:h], v0[h:], zv[h:]
        k_write, v_write = self._mix_kv(kL_w, k0_w, v0_w, zv_w, torch.zeros(1, dtype=torch.long, device=x.device))
        ad = _attn_dtype(v0.dtype)
        if isinstance(history, HPageRead):
            page = (kL_w.detach().to(ad), k0_w.detach().to(ad), v0_w.detach(), zv_w.detach(), x[h:].clone())
        else:
            page = (kL_w.to(ad), k0_w.to(ad), v0_w.clone(), zv_w.clone())
        hist = self._gated_history_checkpointed(history, s, rope_hist)
        if hist is None:
            k_attn, v_attn = torch.cat([k_noisy, k_write]), torch.cat([v_noisy, v_write])
        else:
            k_hist, v_hist = hist
            k_attn = torch.cat([torch.cat([k_hist, k_noisy.to(k_hist.dtype)], dim=1),
                                torch.cat([k_hist, k_write.to(k_hist.dtype)], dim=1)])
            v_attn = torch.cat([torch.cat([v_hist, v_noisy], dim=1), torch.cat([v_hist, v_write], dim=1)])
        y = self.o(_flash_attention(q_rope, k_attn, v_attn).flatten(2))
        return (y, *page)
    # ==== ours end ====
    # ==== ours end ====


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 # ======== ours ========
                 memory_lora=False,
                 memory_lora_distance_gate=None
                 # ==== ours end ====
                 ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps,
                                                # ======== ours ========
                                                memory_lora=memory_lora,
                                                memory_lora_distance_gate=memory_lora_distance_gate
                                                # ==== ours end ====
                                                )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        # ======== ours ========
        persistent=None,
        memory_write=False
        # ==== ours end ====
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # ======== ours ========
        if persistent is not None:
            return self.forward_persistent(x, e, context, **persistent)
        # ==== ours end ====
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start,
            # ======== ours ========
            memory_write=memory_write
            # ==== ours end ====
            )

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


    # ======== ours ========
    def forward_persistent(self, x, e, context, rope, history, write_page=False, kv_only=False,
                           write_mask=None, rope_hist=None):
        """forward() against immutable pages instead of a KV cache, otherwise
        identical. A forward with gradient recomputes cross-attention rather
        than caching it, so the block holds no mutable state that a checkpoint
        recomputation could not replay. With kv_only (the writer's last block,
        whose output nothing reads) only this frame's page is computed and x is
        returned unchanged. With write_mask, x is the fused [noisy; write] batch.
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        out = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            None, None, None, None,
            persistent=dict(rope=rope, history=history, write_page=write_page, kv_only=kv_only,
                            write_mask=write_mask, rope_hist=rope_hist))
        if kv_only:
            return (x, *out)
        y, *page = out if write_page else (out,)     # page: (k, v), or (k, v, h) for h pages
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)
        x = x + self._cross_attn_persistent(self.norm3(x), context, history)
        y = self.ffn(
            (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2))
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
        if write_page:
            return (x, *page)
        return x

    def _cross_attn_persistent(self, x, context, history):
        """self.cross_attn(x, context, None), op for op, with the flash-attn call
        going through _flash_attention. A forward without gradient takes the
        prompt's K/V from the layer's page table, where the rollout's first such
        forward leaves them: they are the same for every frame and denoising
        step, as in the inference path's crossattn_cache."""
        ca = self.cross_attn
        if not isinstance(ca, WanT2VCrossAttention):
            return ca(x, context, None)
        b, n, d = x.size(0), ca.num_heads, ca.head_dim
        q = ca.norm_q(ca.q(x)).view(b, -1, n, d)
        table = None if torch.is_grad_enabled() else history.table
        if table is not None and table.context_kv is not None:
            k, v = table.context_kv
        else:
            k = ca.norm_k(ca.k(context)).view(b, -1, n, d)
            v = ca.v(context).view(b, -1, n, d)
            if table is not None:      # kept, like the pages, in the dtype flash-attn computes in
                k, v = k.to(_attn_dtype(k.dtype)), v.to(_attn_dtype(v.dtype))
                table.context_kv = (k, v)
        return ca.o(_flash_attention(q, k, v).flatten(2))
    # ==== ours end ====


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 # ======== ours ========
                 memory_lora=False,
                 memory_lora_distance_gate=None
                 # ==== ours end ====
                 ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps,
                                    # ======== ours ========
                                    memory_lora=memory_lora,
                                    memory_lora_distance_gate=memory_lora_distance_gate
                                    # ==== ours end ====
                                    )
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False
        # ======== ours ========
        # KV Forcing: checkpoint the two persistent forwards in groups of
        # `persistent_checkpoint_group` blocks (1 = every block). Set by
        # KVForcingTrainingPipeline; inference never takes that path.
        self.persistent_block_checkpointing = False
        self.persistent_checkpoint_group = 1
        # Memory-LoRA distance gate (None = ungated); see MemoryLoRA.gated
        self.memory_lora_distance_gate = memory_lora_distance_gate_mode(memory_lora_distance_gate)
        # ==== ours end ====

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1,
        local_attn_size: int = -1, sink_size: int = 0
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # Block-wise causal mask attends to elements before the end of the current chunk.
        attention_block_size = frame_seqlen * num_frame_per_block

        # Streaming attention = sink + sliding window (StreamingLLM-style), token units.
        #   local_attn_size = -1  -> unbounded recent window (original full behaviour)
        #   local_attn_size  = W  -> attend to the last W frames INCLUDING the current one
        #   sink_size        = S  -> the first S frames are ALWAYS attended (pinned)
        # When local_attn_size == -1 and sink_size == 0 this reduces exactly to the
        # original blockwise-causal teacher-forcing mask.
        # sink/window are measured in LATENTS (frame_seqlen), independent of the
        # chunk size, so chunk=1 and chunk=3 share the same temporal extent. The
        # intra-chunk bidirectional + cross-chunk causal structure still comes from
        # attention_block_size below. sink/window must be multiples of the chunk so
        # the window stays chunk-aligned (matches the chunk-granular cache eviction).
        assert sink_size % num_frame_per_block == 0, \
            f"sink_size ({sink_size}) must be a multiple of num_frame_per_block ({num_frame_per_block})"
        if local_attn_size != -1:
            assert local_attn_size % num_frame_per_block == 0, \
                f"local_attn_size ({local_attn_size}) must be a multiple of num_frame_per_block ({num_frame_per_block})"
        win_tokens = local_attn_size * frame_seqlen if local_attn_size != -1 else None
        sink_tokens = sink_size * frame_seqlen

        # clean context frames: blockwise-causal self-attention in [context_starts, context_ends)
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # noisy frames: recent clean-context window [noise_context_starts, noise_context_ends) + own block
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames (self-attention building their K/V representation)
        for start in frame_indices:
            end = int(start) + attention_block_size
            context_ends[start:start + attention_block_size] = end
            # a clean frame looks back at most `local_attn_size` frames (incl. its own block)
            context_starts[start:start + attention_block_size] = \
                max(0, end - win_tokens) if win_tokens is not None else 0

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # clean context = all frames strictly before this block ...
            ctx_end = block_index * attention_block_size
            noise_context_ends[start:end] = ctx_end
            # ... limited to the most recent (local_attn_size - 1) clean frames
            if win_tokens is not None:
                noise_context_starts[start:end] = max(0, ctx_end - (win_tokens - attention_block_size))
            else:
                noise_context_starts[start:end] = 0

        def attention_mask(b, h, q_idx, kv_idx):
            # clean self-attention: CAUSAL (kv block <= q block) AND in the recent
            # window OR the sink. The causal bound is applied to BOTH terms, so the
            # sink frames themselves stay causal (clean frame 0 -> {0}, 1 -> {0,1}).
            clean_causal = kv_idx < context_ends[q_idx]
            clean_keep = (kv_idx >= context_starts[q_idx]) | (kv_idx < sink_tokens)
            clean_mask = (q_idx < clean_ends) & clean_causal & clean_keep
            # noisy frames: own block (C1) + clean context STRICTLY BEFORE this block
            # (causal), restricted to the recent window OR the sink.
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            ctx_causal = kv_idx < noise_context_ends[q_idx]
            ctx_keep = (kv_idx >= noise_context_starts[q_idx]) | (kv_idx < sink_tokens)
            noise_mask = (q_idx >= clean_ends) & (C1 | (ctx_causal & ctx_keep))

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                                padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        # ======== ours ========
        memory_write: bool = False
        # ==== ours end ====
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        # ======== ours ========
                        "memory_write": memory_write,
                        # ==== ours end ====
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        # ======== ours ========
                        "memory_write": memory_write,
                        # ==== ours end ====
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None: # TF
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                        sink_size=self.sink_size,
                    )
            else: # DF?
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            # clean_x.detach()
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]
            # [1,32760,1536]
        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        # ======== ours ========
        # KV Forcing's two per-frame forwards. Dispatched from forward() so they
        # run through __call__: under FSDP that is where the parameters are
        # gathered and cast; a bare method call would see them still sharded.
        mode = kwargs.pop("persistent_mode", None)
        if mode is not None:
            fn = {"write": self.forward_clean_writer, "noisy": self.forward_noisy_frame,
                  "fused": self.forward_fused_frame}[mode]
            return fn(args[0], kwargs["t"], kwargs["context"],
                      kwargs["persistent_history"], kwargs.get("persistent_start", 0))
        # ==== ours end ====
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            # TF or DF
            return self._forward_train(*args, **kwargs)

    # ======== ours ========
    def _embed_frame(self, latent, t, context):
        """Time/text embedding and the patch grid of one frame forward. The
        patch embedding itself runs inside the first checkpoint region
        (_run_blocks), so its output is not kept between forward and backward."""
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        p = self.patch_size             # kernel = stride, no padding: the grid is size // patch
        grid_sizes = torch.stack([torch.tensor([u.shape[1] // p[0], u.shape[2] // p[1], u.shape[3] // p[2]],
                                               dtype=torch.long) for u in latent])
        e, e0 = self._embed_time(t)
        if torch.is_tensor(context) and context.dim() == 3 and context.size(1) == self.text_len:
            # Already [B, text_len, C]: padding and stacking would only copy it.
            # Used as is, every frame forward of a rollout saves this one tensor
            # for the text embedding's backward instead of its own copy.
            padded = context
        else:
            padded = torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        ctx = self.text_embedding(padded)
        return grid_sizes, e, e0, ctx

    def _embed_time(self, t):
        """Time embedding of t ([B, F]) and its projection onto the blocks' modulation."""
        device = self.patch_embedding.weight.device
        # the dtype of the patch embedding's output (what `type_as(x)` gave)
        dtype = (torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type)
                 else self.patch_embedding.weight.dtype)
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype))
        return e, self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

    def _run_blocks(self, first, last, x, e, e0, ctx, rope, histories, write_page, t_shape,
                    write_mask=None, rope_hist=None):
        """Blocks first..last-1 of one persistent forward, `histories` being
        those blocks' histories. Returns x, then the pages of those blocks
        (k, v per block) when write_page is set. The writer's last block only
        computes its page: nothing reads the writer's output. The region that
        starts at block 0 takes the latent and runs the patch embedding; the
        region that ends at the last block of a noisy forward runs the head:
        neither the embedded frame nor the head's input is then kept between
        forward and backward. With write_mask, x is the fused [noisy; write]
        batch: every block runs in full (the noisy half needs its output) and
        the head takes the noisy half."""
        fused = write_mask is not None
        if first == 0:
            xs = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            x = torch.cat([u.flatten(2).transpose(1, 2) for u in xs])
        pages = []
        for layer_idx, history in zip(range(first, last), histories):
            kv_only = write_page and not fused and layer_idx == len(self.blocks) - 1
            out = self.blocks[layer_idx](x, e0, None, None, None, ctx, None, None,
                                         persistent=dict(rope=rope, history=history,
                                                         write_page=write_page, kv_only=kv_only,
                                                         write_mask=write_mask, rope_hist=rope_hist))
            if write_page:
                x, *page = out
                pages.extend(page)
            else:
                x = out
        if last == len(self.blocks) and (fused or not write_page):
            x = self.head(x[:x.shape[0] // 2] if fused else x, e.unflatten(dim=0, sizes=t_shape).unsqueeze(2))
        return (x, *pages)

    def _persistent_blocks(self, latent, e, e0, grid_sizes, ctx, histories, frame_idx, write_page, t_shape,
                           write_mask=None):
        """One persistent forward from the latent to the head output (noisy) or
        the pages (write), activation-checkpointed in
        groups of `persistent_checkpoint_group` blocks INSIDE the FSDP unit
        (the composition PyTorch supports, as _forward_train does). Only the
        group boundaries are kept between forward and backward: 1 keeps every
        block output, a larger group keeps fewer outputs and recomputes more
        blocks per group. Pages are group outputs, so they are always kept.

        A region is handed the histories of its own blocks only, and does not
        preserve the RNG state. checkpoint() walks every tensor in its
        arguments and, to preserve the RNG state, reads the CUDA generator
        state once per tensor at forward time and reads or sets it three more
        times per tensor at recomputation; the page handles of all 30 layers
        are up to 1200 tensors, for every region. Nothing in a block draws
        random numbers (no dropout; flash-attn with dropout_p=0 takes no
        generator state), so there is no state to preserve."""
        n, g = len(self.blocks), max(1, int(self.persistent_checkpoint_group))
        ckpt = self.persistent_block_checkpointing and torch.is_grad_enabled()
        rope = _rope_table(self.freqs, grid_sizes, frame_idx)
        # h pages: the RoPE table of the history frames 0..frame_idx-1, for the K/V rebuild
        # (distance gate: every forward builds its history K/V and needs it as well)
        rope_hist = (self._history_rope(grid_sizes, frame_idx)
                     if frame_idx and (isinstance(histories[0], HPageRead)
                                       or self.memory_lora_distance_gate is not None) else None)
        x, pages = latent, []
        for first in range(0, n, g):
            last = min(first + g, n)
            args = (first, last, x, e, e0, ctx, rope, histories[first:last], write_page, t_shape,
                    write_mask, rope_hist)
            out = (torch.utils.checkpoint.checkpoint(self._run_blocks, *args, use_reentrant=False,
                                                     preserve_rng_state=False)
                   if ckpt else self._run_blocks(*args))
            x, pages = out[0], pages + list(out[1:])
        return x, pages

    def _history_rope(self, grid_sizes, frames):
        """_rope_table of frames 0..frames-1, as a view of one table kept on the
        model: the forwards of a rollout then share it instead of each keeping
        its own from forward to backward."""
        key = (tuple(grid_sizes[0].tolist()), self.freqs.device)
        tokens = frames * int(grid_sizes[0].prod())
        cached = getattr(self, "_history_rope_cache", None)
        if cached is None or cached[0] != key or cached[1].shape[0] < tokens:
            cached = (key, _rope_table(self.freqs, grid_sizes, 0, frames=frames))
            self._history_rope_cache = cached
        return cached[1][:tokens]

    def forward_noisy_frame(self, latent, t, context, histories, frame_idx):
        """The frame the loss sees: reads the history pages, writes none."""
        grid_sizes, e, e0, ctx = self._embed_frame(latent, t, context)
        x, _ = self._persistent_blocks(latent, e, e0, grid_sizes, ctx, histories, frame_idx,
                                       write_page=False, t_shape=t.shape)
        return torch.stack(self.unpatchify(x, grid_sizes))

    def forward_clean_writer(self, latent, t, context, histories, frame_idx):
        """The memory write. Returns this frame's pages as (k0, v0, k1, v1, ...);
        no head, since a clean frame only provides context."""
        grid_sizes, e, e0, ctx = self._embed_frame(latent, t, context)
        _, pages = self._persistent_blocks(latent, e, e0, grid_sizes, ctx, histories, frame_idx,
                                           write_page=True, t_shape=t.shape)
        return tuple(pages)

    def forward_fused_frame(self, latent, t, context, histories, frame_idx):
        """forward_noisy_frame and forward_clean_writer as ONE forward of the
        batch [noisy; write] (latent and t are the two concatenated; context is
        the unrepeated text). Both halves read the same history and neither
        reads the other's K/V, so every forward, recomputation and backward of
        the pair is one call. Returns (the noisy half's prediction, the write
        half's pages). The halves reproduce the two separate forwards bit for
        bit in the forward: the patch embedding runs per sample, and the token
        GEMMs, norms, RoPE and flash-attn are batch-invariant (checked on H100,
        bf16 and fp32/TF32). Two things are not and are kept per half: the time
        embedding (in fp32 a 1-row GEMM takes another kernel than a 2-row one)
        and the text, embedded once and copied (a stride-0 expanded context
        takes another GEMM path in cross-attention)."""
        h = latent.shape[0] // 2
        grid_sizes, e, e0, ctx = self._embed_frame(latent, t[:h], context)
        e0 = torch.cat([e0, self._embed_time(t[h:])[1]])
        ctx = torch.cat([ctx, ctx])
        write_mask = (torch.arange(2 * h, device=ctx.device) >= h).to(ctx.dtype).view(2 * h, 1, 1)
        x, pages = self._persistent_blocks(latent, e, e0, grid_sizes, ctx, histories, frame_idx,
                                           write_page=True, t_shape=t[:h].shape, write_mask=write_mask)
        return torch.stack(self.unpatchify(x, grid_sizes[:h])), tuple(pages)
    # ==== ours end ====

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
        # ======== ours ========
        # The Xavier pass above hits the adapters' nn.Linear too; dW must start
        # at exactly zero so memory and generation projections begin identical.
        for m in self.modules():
            if isinstance(m, MemoryLoRA):
                m.reset_parameters()
        # ==== ours end ====
