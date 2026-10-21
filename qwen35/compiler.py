"""A small graph compiler for one decode/verify step of Qwen3.5.

    build_graph(cfg)  ->  Graph (IR: embed / rmsnorm / linear / add / swiglu / deltanet / gated_norm /
                          attn_core / head_argmax nodes, weights referenced by name)
    fuse(graph)       ->  pattern-rewrite passes that merge nodes into the fused Metal kernel family
                          (rmsnorm->linear, gated_norm->linear, linear->+residual, linear->swiglu)
    layout(graph, W)  ->  weight-layout pass: materialises the layouts the fused kernels need (gate/up row interleave,
                          lossless 12-bit weight compression) and drops the originals
    lower(graph, ...) ->  static schedule: a straight-line list of kernel calls over a value table,
                          executed under mx.compile
`lower(..., fuse=False)` emits one plain-MLX op per node (the unfused reference) for ablation.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from .head_prune import ENABLED as HEAD_PRUNE, argmax_head, build_tables
from . import wcomp
from .kernels import fused_gemv, deltanet
from .model import Cfg, _write


# --------------------------------------------------------------------------------------------- IR
@dataclass
class Node:
    id: int
    op: str
    ins: list            # node ids
    attrs: dict = field(default_factory=dict)


class Graph:
    def __init__(self):
        self.nodes: list[Node] = []

    def add(self, op, ins=(), **attrs) -> int:
        n = Node(len(self.nodes), op, list(ins), attrs)
        self.nodes.append(n)
        return n.id

    def users(self):
        u = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            for i in n.ins:
                u[i].append(n.id)
        return u

    def live(self):
        """Nodes reachable from outputs (output nodes: op == 'output')."""
        keep, stack = set(), [n.id for n in self.nodes if n.op == "output"]
        while stack:
            i = stack.pop()
            if i in keep:
                continue
            keep.add(i)
            stack.extend(self.nodes[i].ins)
        return [n for n in self.nodes if n.id in keep]

    def dump(self, layers=(0, 3)):
        out = []
        for n in self.live():
            tag = n.attrs.get("layer")
            if tag is None or tag in layers:
                a = {k: v for k, v in n.attrs.items() if k not in ("layer",)}
                out.append(f"  %{n.id:<4d}= {n.op}({', '.join('%' + str(i) for i in n.ins)}) {a if a else ''}  [L{tag}]" if tag is not None
                           else f"  %{n.id:<4d}= {n.op}({', '.join('%' + str(i) for i in n.ins)}) {a if a else ''}")
        return "\n".join(out)


def build_graph(cfg: Cfg) -> Graph:
    g = Graph()
    toks = g.add("input", name="toks")
    x = g.add("embed", [toks])
    for i in range(cfg.layers):
        P = f"layers.{i}."
        h = g.add("rmsnorm", [x], w=P + "ln1", layer=i)
        if cfg.is_linear(i):
            proj = g.add("linear", [h], w=P + "in_proj", layer=i)
            y = g.add("deltanet", [proj], layer=i, w=P)
            yn = g.add("gated_norm", [y, proj], w=P + "gnorm", layer=i)
            r = g.add("linear", [yn], w=P + "out_proj", layer=i)
        else:
            qkv = g.add("linear", [h], w=P + "qkv", layer=i)
            o = g.add("attn_core", [qkv], layer=i, w=P)
            r = g.add("linear", [o], w=P + "o_proj", layer=i)
        x = g.add("add", [x, r], layer=i)
        h2 = g.add("rmsnorm", [x], w=P + "ln2", layer=i)
        gu = g.add("linear", [h2], w=P + "gate_up", layer=i)
        a = g.add("swiglu", [gu], layer=i)
        d = g.add("linear", [a], w=P + "down", layer=i)
        x = g.add("add", [x, d], layer=i)
    hn = g.add("rmsnorm", [x], w="norm", layer="final")
    ids = g.add("head_argmax", [hn], w="embed")
    g.add("output", [ids, hn])
    return g


# ------------------------------------------------------------------------------------------ passes
def fuse(g: Graph, verbose=False) -> Graph:
    """Pattern-rewrite fusion. Each rule only fires when the producer has a single consumer."""
    stats = {"rmsnorm->linear": 0, "gated_norm->linear": 0, "linear->add": 0, "linear->swiglu": 0}
    changed = True
    while changed:
        changed = False
        users = g.users()
        for n in g.live():
            if n.op == "linear":
                src = g.nodes[n.ins[0]]
                if src.op == "rmsnorm" and len(users[src.id]) == 1 and "prologue" not in n.attrs and src.attrs["layer"] != "final":
                    n.attrs["prologue"] = ("rms", src.attrs["w"])
                    n.ins = list(src.ins)
                    stats["rmsnorm->linear"] += 1; changed = True; break
                if src.op == "gated_norm" and len(users[src.id]) == 1 and "prologue" not in n.attrs:
                    n.attrs["prologue"] = ("gnorm", src.attrs["w"])
                    n.ins = [src.ins[0]] + [src.ins[1]]       # (y, proj)
                    stats["gated_norm->linear"] += 1; changed = True; break
            if n.op == "add":
                for k in (0, 1):
                    lin = g.nodes[n.ins[k]]
                    if lin.op == "linear" and len(users[lin.id]) == 1 and "epilogue" not in lin.attrs:
                        res = n.ins[1 - k]
                        lin.attrs["epilogue"] = ("add",)
                        lin.ins = lin.ins + [res]
                        # replace add by the fused linear everywhere
                        for m in g.nodes:
                            m.ins = [lin.id if i == n.id else i for i in m.ins]
                        n.op = "dead"; n.ins = []
                        stats["linear->add"] += 1; changed = True; break
                if changed:
                    break
            if n.op == "swiglu":
                lin = g.nodes[n.ins[0]]
                if lin.op == "linear" and len(users[lin.id]) == 1 and "epilogue" not in lin.attrs:
                    lin.attrs["epilogue"] = ("swiglu",)
                    lin.attrs["layout"] = "interleave_gate_up"
                    for m in g.nodes:
                        m.ins = [lin.id if i == n.id else i for i in m.ins]
                    n.op = "dead"; n.ins = []
                    stats["linear->swiglu"] += 1; changed = True; break
    if verbose:
        print("fusion:", stats)
    g.fusion_stats = stats
    return g


def layout(g: Graph, W: dict) -> dict:
    """Weight-layout pass. W: flat dict name -> array. Returns the dict the schedule reads from."""
    out = dict(W)
    for n in g.live():
        if n.op == "linear" and n.attrs.get("layout") == "interleave_gate_up":
            name = n.attrs["w"]
            gu = out.pop(name)
            h = gu.shape[0] // 2
            out[name] = mx.stack([gu[:h], gu[h:]], axis=1).reshape(2 * h, -1)
    if wcomp.ENABLED:       # every fused linear streams its weight from the lossless 12-bit copy (wcomp.py; bit-identical)
        for n in g.live():
            if n.op == "linear" and not wcomp.is_comp(out[n.attrs["w"]]):
                out[n.attrs["w"]] = wcomp.encode(out[n.attrs["w"]])
    mx.eval(out)
    return out


def flatten_weights(W: dict, cfg: Cfg) -> dict:
    flat = {"embed": W["embed"], "norm": W["norm"]}
    if HEAD_PRUNE:
        flat["head_tab"] = build_tables(W["embed"])
    for i, L in enumerate(W["layers"]):
        for k, v in L.items():
            flat[f"layers.{i}.{k}"] = v
    return flat


# ----------------------------------------------------------------------------------------- lowering
def _attn_core(cfg, Wt, P, qkv, st, ki, pos, kv_len, mask):
    M = qkv.shape[0]
    H, KV, D = cfg.n_heads, cfg.n_kv, cfg.head_dim
    q, k, v = mx.split(qkv, [H * D * 2, H * D * 2 + KV * D], axis=-1)
    q, gate = mx.split(q.reshape(1, M, H, 2 * D), 2, axis=-1)
    gate = gate.reshape(M, -1)
    q = mx.fast.rms_norm(q, Wt[P + "q_norm"], cfg.eps).transpose(0, 2, 1, 3)
    k = mx.fast.rms_norm(k.reshape(1, M, KV, D), Wt[P + "k_norm"], cfg.eps).transpose(0, 2, 1, 3)
    v = v.reshape(1, M, KV, D).transpose(0, 2, 1, 3)
    q = mx.fast.rope(q, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos)
    k = mx.fast.rope(k, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos)
    K_, V_ = st["kv"][ki]
    K_ = _write(K_, k, pos); V_ = _write(V_, v, pos)
    o = mx.fast.scaled_dot_product_attention(q, K_[:, :, :kv_len], V_[:, :, :kv_len], scale=D ** -0.5, mask=mask)
    return o.transpose(0, 2, 1, 3).reshape(M, -1) * mx.sigmoid(gate), (K_, V_)


def lower(g: Graph, cfg: Cfg, fused: bool = True):
    """Return step(Wt, toks, st, sel, pos, kv_len) -> (ids, Hn, new_state) following the graph's static order."""
    order = g.live()
    # per-node static info resolved at lowering time
    li = {i: sum(1 for j in range(i) if cfg.is_linear(j)) for i in range(cfg.layers)}      # conv/ssm index
    ai = {i: sum(1 for j in range(i) if not cfg.is_linear(j)) for i in range(cfg.layers)}  # kv index

    def step(Wt, toks, st, sel, pos, kv_len):
        M = toks.shape[0]
        mask = mx.arange(kv_len)[None, :] <= (pos + mx.arange(M))[:, None]
        v = {}
        new = {"kv": [None] * len(st["kv"]), "conv": [None] * len(st["conv"]), "ssm": [None] * len(st["ssm"])}
        for n in order:
            a = n.attrs
            I = [v[i] for i in n.ins]
            if n.op == "input":
                v[n.id] = toks
            elif n.op == "embed":
                v[n.id] = Wt["embed"][I[0]]
            elif n.op == "rmsnorm":
                v[n.id] = mx.fast.rms_norm(I[0], Wt[a["w"]], cfg.eps)
            elif n.op == "linear":
                W = Wt[a["w"]]
                pro, epi = a.get("prologue"), a.get("epilogue")
                if not fused or (pro is None and epi is None):
                    v[n.id] = fused_gemv(W, I[0]) if fused else I[0] @ W.T
                else:
                    kw = {}
                    if pro and pro[0] == "rms":
                        kw["nw"] = Wt[pro[1]]
                    if pro and pro[0] == "gnorm":
                        kw["gnorm"] = (I[1], Wt[pro[1]])
                        kw.update(T=256, R=8)
                    if epi and epi[0] == "add":
                        kw["res"] = I[-1]
                    if epi and epi[0] == "swiglu":
                        kw["swiglu"] = True
                    v[n.id] = fused_gemv(W, I[0], **kw)
            elif n.op == "add":
                v[n.id] = I[0] + I[1]
            elif n.op == "swiglu":
                gt, u = mx.split(I[0], 2, axis=-1)
                v[n.id] = nn.silu(gt) * u
            elif n.op == "deltanet":
                P, c = a["w"], li[a["layer"]]
                y, cs_, ss_ = deltanet(I[0], st["conv"][c], Wt[P + "conv_w"], Wt[P + "A_log"], Wt[P + "dt_bias"], st["ssm"][c], sel)
                new["conv"][c], new["ssm"][c] = cs_, ss_
                v[n.id] = y
            elif n.op == "gated_norm":
                y, proj = I
                z = proj[:, 6144:8192].astype(mx.float32).reshape(M, 16, 128)
                nrm = mx.fast.rms_norm(y.reshape(M, 16, 128), Wt[a["w"]], cfg.eps).astype(mx.float32)
                v[n.id] = (nn.silu(z) * nrm).astype(mx.bfloat16).reshape(M, 2048)
            elif n.op == "attn_core":
                k_ = ai[a["layer"]]
                o, kv = _attn_core(cfg, Wt, a["w"], I[0], st, k_, pos, kv_len, mask)
                new["kv"][k_] = kv
                v[n.id] = o
            elif n.op == "head_argmax":
                v[n.id] = argmax_head(Wt["embed"], Wt.get("head_tab"), I[0])
            elif n.op == "output":
                return v[n.ins[0]], v[n.ins[1]], new
            else:
                raise NotImplementedError(n.op)
    return step


class Compiled:
    """build -> (fuse) -> layout -> lower, with per-kv-bucket mx.compile'd steps."""

    def __init__(self, W, cfg: Cfg, fuse_pass=True, compile=True):
        self.cfg = cfg
        self.graph = build_graph(cfg)
        if fuse_pass:
            fuse(self.graph)
        self.Wt = layout(self.graph, flatten_weights(W, cfg)) if fuse_pass else flatten_weights(W, cfg)
        self.step_fn = lower(self.graph, cfg, fused=fuse_pass)
        self.compile = compile
        self._c = {}

    def n_kernels(self):
        return sum(1 for n in self.graph.live() if n.op not in ("input", "output"))

    def verify(self, toks, st, sel, pos, kv_len):
        f = self._c.get(kv_len)
        if f is None:
            fn = lambda Wt, t, s, sl, p: self.step_fn(Wt, t, s, sl, p, kv_len)
            f = self._c[kv_len] = mx.compile(fn) if self.compile else fn
        return f(self.Wt, toks, st, sel, pos)
