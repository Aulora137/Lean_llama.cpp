#!/usr/bin/env python3
"""ATTACK 1c (deployability): CROSS-CORPUS transfer of the fitted adapter.

The within-eval xval fit is on tokens adjacent to the test tokens -> optimistic.
The deployable question: fit the rank-r adapter on gemma3 (donor corpus), apply
UNCHANGED to the disjoint xval wikitext dump (0 shared 8-grams), evaluate xval
softmax-KL.  Same transfer protocol as the study's rule-5 donor test.

Arms on xval (all eval-half softmax-KL vs FP):
  tq3           : scalar 3-bit
  ceil_donor    : A=B=U_r, U_r = gemma3-CALIB query eigenbasis (study's ceiling,
                  transferred) -- the study's own cross-corpus ceiling
  fit_sym_donor : A=B fit on gemma3 (softmax-KL), transferred to xval
If fit_sym_donor reaches TQ3 at small r on xval, the adapter is deployable and
BEATS TQ3 at lower storage (r=16 int4 = 2.81 bpe << 3.5). If it degrades to
>=TQ3, the within-eval win was corpus-specific optimism.
"""
from __future__ import annotations
import sys, time, json, argparse
from pathlib import Path
import numpy as np
KIT = Path("/home/junc/Lean_llama.cpp/kit-v2")
sys.path.insert(0, str(KIT))
import contour_study as cs             # noqa
import softmax_correction_study as sc  # noqa
import torch                            # noqa
torch.set_num_threads(6)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
SCRATCH = Path("/tmp/claude-1000/-home-junc-rikuri-rikurinode/"
               "0a7b0d58-b4e2-4aa3-bb15-3e35bc56022f/scratchpad")


def masked_kl(P_ref, logP_ref, L, vist):
    neg = torch.finfo(L.dtype).min
    Lm = torch.where(vist, L, torch.full_like(L, neg))
    logQ = Lm - torch.logsumexp(Lm, -1, keepdim=True)
    return (P_ref * (logP_ref - logQ)).sum(-1)


def eig_basis(Qrows):
    C = (Qrows.double().T @ Qrows.double()) / Qrows.shape[0]
    lam, U = torch.linalg.eigh(C)
    idx = torch.argsort(lam, descending=True)
    return U[:, idx].float()


def head_tensors(K, Q, kq2, il, cfg, h, qh_to_kv, which):
    """Return Qe[g,Ne,hd], K_h,kq2_h,ke_h [T,hd], vis_e[Ne,T], scale, P_ref,logP_ref.
    which='eval' uses eval-half query rows; 'all' uses all tokens as queries."""
    T, nkv, hd = K.shape
    _, nqh, _ = Q.shape
    group = nqh // nkv
    scale = 1.0 if cfg["scale"] == "one" else 1.0 / np.sqrt(hd)
    Tc = T // 2
    tpos = np.arange(T)
    qsel = slice(Tc, T) if which == "eval" else slice(0, T)
    qpos = tpos[qsel]
    swa = None
    if cfg["swa_window"] is not None and not cfg["swa_global"](il):
        swa = cfg["swa_window"]
    vis = qpos[:, None] >= tpos[None, :]
    if swa is not None and swa < T:
        vis &= (qpos[:, None] - tpos[None, :]) < swa
    qidx = np.where(qh_to_kv == h)[0]
    Qe = torch.from_numpy(np.ascontiguousarray(
        Q[qsel, qidx, :].transpose(1, 0, 2))).float()
    K_h = torch.from_numpy(np.ascontiguousarray(K[:, h, :])).float()
    kq2_h = torch.from_numpy(np.ascontiguousarray(kq2[:, h, :])).float()
    ke_h = K_h - kq2_h
    vist = torch.from_numpy(vis)
    L_fp = torch.einsum("gtd,sd->gts", Qe, K_h) * scale
    neg = torch.finfo(L_fp.dtype).min
    Lm = torch.where(vist[None], L_fp, torch.full_like(L_fp, neg))
    logP_ref = Lm - torch.logsumexp(Lm, -1, keepdim=True)
    P_ref = logP_ref.exp()
    L_base = torch.einsum("gtd,sd->gts", Qe, kq2_h)
    return dict(Qe=Qe, ke_h=ke_h, kq2_h=kq2_h, K_h=K_h, vist=vist, scale=scale,
                P_ref=P_ref, logP_ref=logP_ref, L_base=L_base, Tc=Tc, qidx=qidx)


def run(layers, r_list, steps, lr):
    gcfg = next(c for c in sc.MODELS if c["name"] == "gemma3-4b")
    xcfg = next(c for c in sc.MODELS if c["name"] == "gemma3-4b/xval")
    gK = cs.read_kcal_layers(sc.ROOT / gcfg["kf"]); gQ = cs.read_kcal_layers(sc.ROOT / gcfg["qf"])
    xK = cs.read_kcal_layers(sc.ROOT / xcfg["kf"]); xQ = cs.read_kcal_layers(sc.ROOT / xcfg["qf"])
    all_layers = sorted(gK)
    if layers:
        all_layers = [all_layers[i] for i in layers]
    max_il = max(sorted(gK)) + 1
    tqc = {}
    for il in all_layers:
        hd = gK[il].shape[2]
        for b in (2, 3):
            if (hd, b) not in tqc:
                tqc[(hd, b)] = cs.TurboQuantizer(
                    n_layers=max_il, head_dim=hd, bits=b, group_size=None,
                    rotation_strategy="randomized_hadamard", use_qjl=False,
                    seed=42, device="cpu")
    print(f"== gemma3 -> xval CROSS-CORPUS == layers {all_layers} r {r_list}", flush=True)
    rows = []
    t0 = time.time()
    for il in all_layers:
        hd = gK[il].shape[2]; nkv = gK[il].shape[1]
        nqh = gQ[il].shape[1]; group = nqh // nkv
        qh_to_kv = np.arange(nqh) // group
        gkq2 = sc.tq_khat(gK[il], il, tqc[(hd, 2)])
        xkq2 = sc.tq_khat(xK[il], il, tqc[(hd, 2)])
        xkq3 = sc.tq_khat(xK[il], il, tqc[(hd, 3)])
        # xval TQ3 eval KL (study path)
        xrefs = sc.build_refs(xK[il], xQ[il], il, xcfg)
        tq3 = sc.eval_khat(xkq3, xQ[il], xrefs)["kl"]
        for r in r_list:
            ph = {"ceil_donor": [], "fit_sym_donor": [], "ceil_selfcal": []}
            for h in range(nkv):
                # fit on gemma3 (donor): use ALL gemma3 tokens as queries
                gd = head_tensors(gK[il], gQ[il], gkq2, il, gcfg, h, qh_to_kv, "all")
                # gemma3 calib eigbasis (study ceiling subspace)
                Tc = gd["Tc"]
                Qcal = torch.from_numpy(np.ascontiguousarray(
                    gQ[il][:Tc, gd["qidx"], :].reshape(-1, hd))).float()
                U_cal = eig_basis(Qcal)[:, :r].T.contiguous()
                # fit symmetric adapter on gemma3 (all tokens)
                A = torch.nn.Parameter(U_cal.clone())
                opt = torch.optim.Adam([A], lr=lr)
                for _ in range(steps):
                    opt.zero_grad()
                    aq = torch.einsum("gtd,rd->gtr", gd["Qe"], A)
                    bk = gd["ke_h"] @ A.T
                    delta = torch.einsum("gtr,sr->gts", aq, bk)
                    L = (gd["L_base"] + delta) * gd["scale"]
                    loss = masked_kl(gd["P_ref"], gd["logP_ref"], L, gd["vist"]).mean()
                    loss.backward(); opt.step()
                A = A.detach()
                # apply to xval eval half
                xd = head_tensors(xK[il], xQ[il], xkq2, il, xcfg, h, qh_to_kv, "eval")
                # xval self-calib eigbasis (upper-ref: xval's own calib subspace)
                Qxcal = torch.from_numpy(np.ascontiguousarray(
                    xQ[il][:xd["Tc"], xd["qidx"], :].reshape(-1, hd))).float()
                U_xcal = eig_basis(Qxcal)[:, :r].T.contiguous()

                def xkl(M):
                    aq = torch.einsum("gtd,rd->gtr", xd["Qe"], M)
                    bk = xd["ke_h"] @ M.T
                    delta = torch.einsum("gtr,sr->gts", aq, bk)
                    L = (xd["L_base"] + delta) * xd["scale"]
                    return float(masked_kl(xd["P_ref"], xd["logP_ref"], L, xd["vist"]).mean())
                ph["ceil_donor"].append(xkl(U_cal))
                ph["ceil_selfcal"].append(xkl(U_xcal))
                ph["fit_sym_donor"].append(xkl(A))
            rec = dict(il=il, r=r, hd=hd, tq3=tq3,
                       **{k: float(np.mean(v)) for k, v in ph.items()})
            rows.append(rec)
            print(f"  L{il:2d} r={r:3d} xTQ3={tq3:.4f} ceil_donor={rec['ceil_donor']:.4f} "
                  f"ceil_selfcal={rec['ceil_selfcal']:.4f} fit_sym_donor={rec['fit_sym_donor']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    out = SCRATCH / "attack1c_xval.json"
    out.write_text(json.dumps(rows))
    print(f"\n== gemma3->xval CROSS-CORPUS AGG (mean {len(all_layers)} layers) ==")
    print(f"{'r':>4} {'xTQ3':>8} {'ceil_donor':>11} {'ceil_selfcal':>12} "
          f"{'fit_sym_donor':>13} {'fit<TQ3?':>9}")
    for r in r_list:
        rr = [x for x in rows if x["r"] == r]
        m = lambda k: float(np.mean([x[k] for x in rr]))
        beat = "YES" if m("fit_sym_donor") <= m("tq3") else "no"
        print(f"{r:>4} {m('tq3'):>8.4f} {m('ceil_donor'):>11.4f} {m('ceil_selfcal'):>12.4f} "
              f"{m('fit_sym_donor'):>13.4f} {beat:>9}")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="")
    ap.add_argument("--r", default="8,16,32,64")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",") if x != ""] if a.layers else None
    run(layers, [int(x) for x in a.r.split(",")], a.steps, a.lr)
