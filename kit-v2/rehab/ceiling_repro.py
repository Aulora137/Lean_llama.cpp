#!/usr/bin/env python3
"""Harness-faithfulness check: reproduce the study's published ceiling curve
(top-r calib query-eigenbasis, TRUE fp coeffs, FULL-K eval softmax-KL) over ALL
layers, so my Attack-1 ceiling is bit-comparable to the doc."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
KIT = Path("/home/junc/Lean_llama.cpp/kit-v2")
sys.path.insert(0, str(KIT))
import contour_study as cs             # noqa
import softmax_correction_study as sc  # noqa
import torch                            # noqa
torch.set_num_threads(6)


def run(short, r_list):
    cfg = next(c for c in sc.MODELS if c["name"] == sc.SHORT[short])
    Ks = cs.read_kcal_layers(sc.ROOT / cfg["kf"])
    Qs = cs.read_kcal_layers(sc.ROOT / cfg["qf"])
    layers = sorted(Ks)
    max_il = max(layers) + 1
    tqc = {}
    for il in layers:
        hd = Ks[il].shape[2]
        for b in (2, 3, 4):
            if (hd, b) not in tqc:
                tqc[(hd, b)] = cs.TurboQuantizer(
                    n_layers=max_il, head_dim=hd, bits=b, group_size=None,
                    rotation_strategy="randomized_hadamard", use_qjl=False,
                    seed=42, device="cpu")
    acc = {f"ceil{r}": [] for r in r_list}
    for k in ("tq2", "tq3", "tq4"):
        acc[k] = []
    for il in layers:
        K, Q = Ks[il], Qs[il]
        T, nkv, hd = K.shape
        refs = sc.build_refs(K, Q, il, cfg)
        Tc, scale, qh_to_kv = refs["Tc"], refs["scale"], refs["qh_to_kv"]
        for b, nm in ((2, "tq2"), (3, "tq3"), (4, "tq4")):
            acc[nm].append(sc.eval_khat(sc.tq_khat(K, il, tqc[(hd, b)]), Q, refs)["kl"])
        kq = sc.tq_khat(K, il, tqc[(hd, 2)])
        ke = (K - kq).astype(np.float32)
        Qe = Q[Tc:]
        Lb_raw = np.einsum("thd,shd->hts", Qe, kq[:, qh_to_kv, :], optimize=True)
        U = []
        for h in range(nkv):
            qidx = np.where(qh_to_kv == h)[0]
            Qcal = torch.from_numpy(np.ascontiguousarray(Q[:Tc, qidx, :].reshape(-1, hd)))
            Uh, _ = sc.query_basis(Qcal)
            U.append(Uh.numpy().astype(np.float32))
        for r in r_list:
            if r > hd:
                acc[f"ceil{r}"].append(np.nan); continue
            L_corr = np.zeros_like(Lb_raw)
            for h in range(nkv):
                qidx = np.where(qh_to_kv == h)[0]
                Ur = U[h][:, :r]
                c = ke[:, h, :] @ Ur                    # [T,r] true coeffs
                Qe_h = np.ascontiguousarray(Qe[:, qidx, :].transpose(1, 0, 2))
                Qp = np.einsum("gtd,dr->gtr", Qe_h, Ur, optimize=True)
                L_corr[qidx] = np.einsum("gtr,sr->gts", Qp, c, optimize=True)
            L = (Lb_raw + L_corr) * scale
            acc[f"ceil{r}"].append(sc.kl_from_logits(L, refs)["kl"])
    print(f"== {cfg['name']} ({len(layers)} layers) harness-repro ==")
    m = lambda k: float(np.nanmean(acc[k]))
    print(f"  TQ2={m('tq2'):.4f} TQ3={m('tq3'):.4f} TQ4={m('tq4'):.4f}")
    for r in r_list:
        print(f"  ceiling r={r:3d}: {m(f'ceil{r}'):.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True)
    ap.add_argument("--r", default="8,16,32,64,128")
    a = ap.parse_args()
    run(a.short, [int(x) for x in a.r.split(",")])
