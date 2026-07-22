#!/usr/bin/env python3
"""ATTACK 1 (decisive): actually FIT a rank-r linear adapter that optimizes the
ATTENTION objective (softmax-KL to FP), on CALIB queries, evaluate on EVAL.

Compare, per (arch, r):
  - ceiling  = top-r query-eigenbasis correction with TRUE coeffs (== study's
               ceiling; == adapter at init, step 0)
  - fitted   = A,B in R[r,hd] fit by GD to minimise softmax-KL(FP||corrected)
               on CALIB queries, early-stopped on an inner calib holdout,
               evaluated on the EVAL half.
  - TQ3      = scalar 3-bit (the bar the study says the ceiling cannot reach at
               small r).

If a small-r fitted adapter reaches TQ3 (or clearly beats the ceiling at the
same r) on EVAL, the "ceiling bounds the adapter" claim is FALSE.

Reuses the study harness verbatim (softmax_correction_study.build_refs / tq_khat
/ query_basis / kl_from_logits) so the ceiling reproduces the published numbers.
Does NOT modify any study file.
"""
from __future__ import annotations
import sys, time, json, argparse
from pathlib import Path
import numpy as np

KIT = Path("/home/junc/Lean_llama.cpp/kit-v2")
sys.path.insert(0, str(KIT))
import contour_study as cs          # noqa
import softmax_correction_study as sc  # noqa
import torch                         # noqa

torch.set_num_threads(6)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

SCRATCH = Path("/tmp/claude-1000/-home-junc-rikuri-rikurinode/"
               "0a7b0d58-b4e2-4aa3-bb15-3e35bc56022f/scratchpad")


def calib_pack(K, Q, il, cfg):
    """Build per-kv-head CALIB fitting tensors + FP target softmax."""
    T, nkv, hd = K.shape
    _, nqh, _ = Q.shape
    group = nqh // nkv
    scale = 1.0 if cfg["scale"] == "one" else 1.0 / np.sqrt(hd)
    Tc = T // 2
    tpos = np.arange(Tc)
    qh_to_kv = np.arange(nqh) // group
    swa = None
    if cfg["swa_window"] is not None and not cfg["swa_global"](il):
        swa = cfg["swa_window"]
    vis = tpos[:, None] >= tpos[None, :]
    if swa is not None and swa < T:
        vis &= (tpos[:, None] - tpos[None, :]) < swa
    return dict(Tc=Tc, scale=scale, group=group, qh_to_kv=qh_to_kv,
                vis=vis, hd=hd, nkv=nkv, nqh=nqh)


def masked_kl_torch(P_ref, logP_ref, L, vis_t):
    """KL(P_ref || softmax(L)) mean over rows.  L [.,Ne,T], vis_t bool [Ne,T]."""
    neg = torch.finfo(L.dtype).min
    Lm = torch.where(vis_t, L, torch.full_like(L, neg))
    logQ = Lm - torch.logsumexp(Lm, dim=-1, keepdim=True)
    kl = (P_ref * (logP_ref - logQ)).sum(-1)          # rows
    # only rows with >=1 valid key contribute; P_ref rows all sum to 1
    return kl.mean()


def fit_head(Qc_g, Kc, kqc, kec, vis, scale, r, U_r, steps, lr, wd, val_frac,
             seed=0):
    """Fit A,B [r,hd] on calib for one kv-head.  Returns (A,B) np.float32.
    Qc_g [group,Tc,hd]; Kc/kqc/kec [Tc,hd] (calib keys); vis bool [Tc,Tc]."""
    dev = "cpu"
    Qc = torch.from_numpy(Qc_g).float()
    Kc_t = torch.from_numpy(Kc).float()
    kqc_t = torch.from_numpy(kqc).float()
    kec_t = torch.from_numpy(kec).float()
    vist = torch.from_numpy(vis)
    group, Tc, hd = Qc.shape
    # FP + base raw logits (shared)
    L_fp = torch.einsum("gtd,sd->gts", Qc, Kc_t) * scale          # [g,Tc,Tc]
    L_base = torch.einsum("gtd,sd->gts", Qc, kqc_t)               # raw
    neg = torch.finfo(L_fp.dtype).min
    Lm = torch.where(vist[None], L_fp, torch.full_like(L_fp, neg))
    logP_ref = Lm - torch.logsumexp(Lm, -1, keepdim=True)
    P_ref = logP_ref.exp()

    # inner train/val split on query token positions
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(Tc, generator=g)
    n_val = max(1, int(round(Tc * val_frac)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    A = torch.nn.Parameter(torch.from_numpy(U_r.copy()).float())  # [r,hd]
    B = torch.nn.Parameter(torch.from_numpy(U_r.copy()).float())
    opt = torch.optim.Adam([A, B], lr=lr, weight_decay=wd)

    def loss_on(idx):
        aq = torch.einsum("gtd,rd->gtr", Qc[:, idx, :], A)        # [g,ni,r]
        bk = kec_t @ B.T                                          # [Tc,r]
        delta = torch.einsum("gtr,sr->gts", aq, bk)              # [g,ni,Tc]
        L = (L_base[:, idx, :] + delta) * scale
        return masked_kl_torch(P_ref[:, idx, :], logP_ref[:, idx, :],
                               L, vist[idx])

    best_val = float("inf")
    best = (A.detach().clone(), B.detach().clone())
    patience, bad = max(20, steps // 6), 0
    for step in range(steps):
        opt.zero_grad()
        l = loss_on(tr_idx)
        l.backward()
        opt.step()
        with torch.no_grad():
            v = float(loss_on(val_idx))
        if v < best_val - 1e-6:
            best_val = v
            best = (A.detach().clone(), B.detach().clone())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best[0].numpy(), best[1].numpy(), best_val


def fit_head_oracle(Qe_g, K_all, kq_all, ke_all, vis_e, scale, r, U_r,
                    steps, lr):
    """ORACLE fit: fit A,B on the EVAL queries directly (in-sample), the tightest
    upper bound on any rank-r softmax-KL adapter.  Qe_g [group,Ne,hd];
    K_all/kq_all/ke_all [T,hd]; vis_e bool [Ne,T]."""
    Qe = torch.from_numpy(Qe_g).float()
    K_t = torch.from_numpy(K_all).float()
    kq_t = torch.from_numpy(kq_all).float()
    ke_t = torch.from_numpy(ke_all).float()
    vist = torch.from_numpy(vis_e)
    L_fp = torch.einsum("gtd,sd->gts", Qe, K_t) * scale
    L_base = torch.einsum("gtd,sd->gts", Qe, kq_t)
    neg = torch.finfo(L_fp.dtype).min
    Lm = torch.where(vist[None], L_fp, torch.full_like(L_fp, neg))
    logP_ref = Lm - torch.logsumexp(Lm, -1, keepdim=True)
    P_ref = logP_ref.exp()
    A = torch.nn.Parameter(torch.from_numpy(U_r.copy()).float())
    B = torch.nn.Parameter(torch.from_numpy(U_r.copy()).float())
    opt = torch.optim.Adam([A, B], lr=lr)
    bk_const = None
    best = float("inf")
    best_ab = (A.detach().clone(), B.detach().clone())
    for step in range(steps):
        opt.zero_grad()
        aq = torch.einsum("gtd,rd->gtr", Qe, A)
        bk = ke_t @ B.T
        delta = torch.einsum("gtr,sr->gts", aq, bk)
        L = (L_base + delta) * scale
        loss = masked_kl_torch(P_ref, logP_ref, L, vist)
        loss.backward()
        opt.step()
        lv = float(loss)
        if lv < best - 1e-7:
            best = lv
            best_ab = (A.detach().clone(), B.detach().clone())
    return best_ab[0].numpy(), best_ab[1].numpy(), best


def eval_layer_adapter(K, Q, il, cfg, tqc, r, AB_heads):
    """Assemble full-layer corrected logits from per-head (A,B) and return the
    study's EVAL softmax-KL (via sc.kl_from_logits).  AB_heads[h]=(A,B) or None
    for ceiling (uses top-r eigbasis via A=B=U_r)."""
    refs = sc.build_refs(K, Q, il, cfg)
    T, nkv, hd = K.shape
    Tc, scale, qh_to_kv = refs["Tc"], refs["scale"], refs["qh_to_kv"]
    kq = sc.tq_khat(K, il, tqc[(hd, 2)])
    ke = (K - kq).astype(np.float32)
    Qe = Q[Tc:]
    Lb_raw = np.einsum("thd,shd->hts", Qe, kq[:, qh_to_kv, :], optimize=True)
    L_corr = np.zeros_like(Lb_raw)
    for h, AB in enumerate(AB_heads):
        A, B = AB
        qidx = np.where(qh_to_kv == h)[0]
        Qe_h = np.ascontiguousarray(Qe[:, qidx, :].transpose(1, 0, 2))  # [g,Ne,hd]
        aq = np.einsum("gtd,rd->gtr", Qe_h, A, optimize=True)
        bk = ke[:, h, :] @ B.T                                          # [T,r]
        Lc = np.einsum("gtr,sr->gts", aq, bk, optimize=True)
        L_corr[qidx] = Lc
    L = (Lb_raw + L_corr) * scale
    return sc.kl_from_logits(L, refs)["kl"]


def run(short, layers, r_list, steps, lr, wd, val_frac, mode):
    cfg = next(c for c in sc.MODELS if c["name"] == sc.SHORT[short])
    Ks = cs.read_kcal_layers(sc.ROOT / cfg["kf"])
    Qs = cs.read_kcal_layers(sc.ROOT / cfg["qf"])
    all_layers = sorted(Ks)
    if layers:
        all_layers = [all_layers[i] for i in layers] if isinstance(layers[0], int) and max(layers) < len(all_layers) else layers
    max_il = max(sorted(Ks)) + 1
    tqc = {}
    for il in all_layers:
        hd = Ks[il].shape[2]
        for b in (2, 3, 4):
            if (hd, b) not in tqc:
                tqc[(hd, b)] = cs.TurboQuantizer(
                    n_layers=max_il, head_dim=hd, bits=b, group_size=None,
                    rotation_strategy="randomized_hadamard", use_qjl=False,
                    seed=42, device="cpu")
    print(f"== {cfg['name']} == layers {all_layers} r {r_list}", flush=True)
    rows = []
    t0 = time.time()
    for il in all_layers:
        K, Q = Ks[il], Qs[il]
        hd = K.shape[2]
        cp = calib_pack(K, Q, il, cfg)
        Tc, group, qh_to_kv = cp["Tc"], cp["group"], cp["qh_to_kv"]
        nkv, scale, vis = cp["nkv"], cp["scale"], cp["vis"]
        refs = sc.build_refs(K, Q, il, cfg)
        kq = sc.tq_khat(K, il, tqc[(hd, 2)])
        ke = (K - kq).astype(np.float32)
        # TQ2/TQ3/TQ4 eval KL (study path)
        tq_kl = {}
        for b in (2, 3, 4):
            kh = sc.tq_khat(K, il, tqc[(hd, b)])
            tq_kl[b] = sc.eval_khat(kh, Q, refs)["kl"]
        # per-head calib packs + eigbasis
        U_heads = []
        for h in range(nkv):
            qidx = np.where(qh_to_kv == h)[0]
            Qcal = torch.from_numpy(np.ascontiguousarray(
                Q[:Tc, qidx, :].reshape(-1, hd)))
            U, _ = sc.query_basis(Qcal)                 # [hd,hd] desc
            U_heads.append(U.numpy().astype(np.float32))
        # eval-side masks for oracle fit
        refs_e = refs
        T = K.shape[0]
        tposT = np.arange(T)
        epos = tposT[Tc:]
        swa = None
        if cfg["swa_window"] is not None and not cfg["swa_global"](il):
            swa = cfg["swa_window"]
        vis_e = epos[:, None] >= tposT[None, :]
        if swa is not None and swa < T:
            vis_e &= (epos[:, None] - tposT[None, :]) < swa
        for r in r_list:
            ceil_heads, fit_heads, orc_heads = [], [], []
            calib_val = []
            for h in range(nkv):
                qidx = np.where(qh_to_kv == h)[0]
                U_r = U_heads[h][:, :r].T.copy()        # [r,hd]
                ceil_heads.append((U_r, U_r))
                if mode in ("honest", "both"):
                    Qc_g = np.ascontiguousarray(
                        Q[:Tc, qidx, :].transpose(1, 0, 2))
                    Kc = np.ascontiguousarray(K[:Tc, h, :])
                    kqc = np.ascontiguousarray(kq[:Tc, h, :])
                    kec = np.ascontiguousarray(ke[:Tc, h, :])
                    A, B, bv = fit_head(Qc_g, Kc, kqc, kec, vis, scale, r, U_r,
                                        steps, lr, wd, val_frac)
                    fit_heads.append((A, B))
                    calib_val.append(bv)
                if mode in ("oracle", "both"):
                    Qe_g = np.ascontiguousarray(
                        Q[Tc:, qidx, :].transpose(1, 0, 2))
                    Ka = np.ascontiguousarray(K[:, h, :])
                    kqa = np.ascontiguousarray(kq[:, h, :])
                    kea = np.ascontiguousarray(ke[:, h, :])
                    Ao, Bo, bo = fit_head_oracle(Qe_g, Ka, kqa, kea, vis_e,
                                                 scale, r, U_r, steps, lr)
                    orc_heads.append((Ao, Bo))
            kl_ceil = eval_layer_adapter(K, Q, il, cfg, tqc, r, ceil_heads)
            kl_fit = eval_layer_adapter(K, Q, il, cfg, tqc, r, fit_heads) \
                if fit_heads else float("nan")
            kl_orc = eval_layer_adapter(K, Q, il, cfg, tqc, r, orc_heads) \
                if orc_heads else float("nan")
            rows.append(dict(il=il, r=r, hd=hd, tq2=tq_kl[2], tq3=tq_kl[3],
                             tq4=tq_kl[4], ceil=kl_ceil, fit=kl_fit,
                             orc=kl_orc,
                             calib_val=float(np.mean(calib_val))
                             if calib_val else float("nan")))
            print(f"  L{il:2d} r={r:3d} hd={hd} TQ3={tq_kl[3]:.4f} "
                  f"ceil={kl_ceil:.4f} honest={kl_fit:.4f} oracle={kl_orc:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    out = SCRATCH / f"attack1_{short}_{mode}.json"
    out.write_text(json.dumps(rows))
    # aggregate
    print(f"\n== {cfg['name']} AGGREGATE (mean over {len(all_layers)} layers, "
          f"mode={mode}) ==")
    print(f"{'r':>4} {'TQ2':>8} {'TQ3':>8} {'TQ4':>8} {'ceiling':>8} "
          f"{'honest':>8} {'oracle':>8} {'orc<TQ3?':>9}")
    for r in r_list:
        rr = [x for x in rows if x["r"] == r]
        m = lambda k: float(np.nanmean([x[k] for x in rr]))
        beat = "YES" if m("orc") <= m("tq3") else "no"
        print(f"{r:>4} {m('tq2'):>8.4f} {m('tq3'):>8.4f} {m('tq4'):>8.4f} "
              f"{m('ceil'):>8.4f} {m('fit'):>8.4f} {m('orc'):>8.4f} {beat:>9}")
    print(f"wrote {out}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True)
    ap.add_argument("--layers", default="")   # comma list of INDICES into sorted layers
    ap.add_argument("--r", default="8,16,32")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--mode", default="both", choices=["honest", "oracle", "both"])
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",") if x != ""] if a.layers else None
    r_list = [int(x) for x in a.r.split(",")]
    run(a.short, layers, r_list, a.steps, a.lr, a.wd, a.val_frac, a.mode)
