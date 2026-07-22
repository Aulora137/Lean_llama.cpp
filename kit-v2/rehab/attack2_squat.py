#!/usr/bin/env python3
"""ATTACK 2: reconcile the softmax-correction study's "SQuat-3bit beats TQ3 by
~2.7x" (0.032/0.034/0.045 vs TQ3 0.055/0.088/0.088) against the committed
noiseshape doc ("subspace shaping only +8-14 pts additive over free mse_opt").

Decompose the win, all FULL-K, all with the study's identical eval path
(sc.eval_khat), calib/eval split exactly as the studies:

  TQ3 (TurboQuant)  = sc.tq_khat, bits=3  == study's "scalar TQ3" bar (0.0885)
  amax-3b (ns codec)= plain 3-bit in the SHAPING codec (Hadamard+amax+TQ levels)
  mse-3b  (ns codec)= free per-block mse_opt scale, NO subspace (arm B)
  SQuat-3b calib    = greedy shaping, subspace fit on CALIB queries (honest)
  SQuat-3b ORACLE   = subspace fit on EVAL queries (peeks) -> oracle inflation
  (same ladder at 2 bits to connect to the noiseshape "additive" claim)

If SQuat's win over TQ3 is mostly (codec + free mse_opt scale) with only a small
subspace increment, and the calib/oracle gap is small, then the number is HONEST
but MISLABELLED as "shaping": noiseshape's conclusion stands.
"""
from __future__ import annotations
import sys, time, json, argparse
from pathlib import Path
import numpy as np

KIT = Path("/home/junc/Lean_llama.cpp/kit-v2")
sys.path.insert(0, str(KIT))
import contour_study as cs           # noqa
import noiseshape_study as ns        # noqa
import softmax_correction_study as sc  # noqa
import torch                          # noqa

torch.set_num_threads(6)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
SCRATCH = Path("/tmp/claude-1000/-home-junc-rikuri-rikurinode/"
               "0a7b0d58-b4e2-4aa3-bb15-3e35bc56022f/scratchpad")
SHAPE_R, SHAPE_LAM = 32, 1.0


def squat_khat(Xr_h, Qsrc_rot, R, hd, bits, sch):
    """SQuat greedy shaped quant for one head. Qsrc_rot [Nq,hd] rotated queries
    (calib or eval) used to fit the subspace. Returns Khat_h [T,hd] (unrotated)."""
    Qhat = ns.squat_subspace(Qsrc_rot, SHAPE_R)
    ps = ns.correction_vectors(Qhat, SHAPE_LAM)
    d32 = ns.block_scale(Xr_h, hd, bits, sch)
    return (ns.greedy_quant(Xr_h, d32, bits, ps) @ R).numpy()


def run(short, layers):
    cfg = next(c for c in sc.MODELS if c["name"] == sc.SHORT[short])
    Ks = cs.read_kcal_layers(sc.ROOT / cfg["kf"])
    Qs = cs.read_kcal_layers(sc.ROOT / cfg["qf"])
    all_layers = sorted(Ks)
    if layers:
        all_layers = [all_layers[i] for i in layers]
    max_il = max(sorted(Ks)) + 1
    tqc, rot = {}, {}
    for il in all_layers:
        hd = Ks[il].shape[2]
        for b in (2, 3, 4):
            if (hd, b) not in tqc:
                tqc[(hd, b)] = cs.TurboQuantizer(
                    n_layers=max_il, head_dim=hd, bits=b, group_size=None,
                    rotation_strategy="randomized_hadamard", use_qjl=False,
                    seed=42, device="cpu")
        if hd not in rot:
            rot[hd] = tqc[(hd, 2)].rotations
    print(f"== {cfg['name']} == layers {all_layers}", flush=True)
    rows = []
    t0 = time.time()
    for il in all_layers:
        K, Q = Ks[il], Qs[il]
        T, nkv, hd = K.shape
        _, nqh, _ = Q.shape
        group = nqh // nkv
        Tc = T // 2
        qh_to_kv = np.arange(nqh) // group
        refs = sc.build_refs(K, Q, il, cfg)
        R = rot[hd][il].float()
        rec = dict(il=il, hd=hd)
        # TurboQuant ladder (== study scalar TQ2/TQ3)
        for b in (2, 3):
            rec[f"tq{b}"] = sc.eval_khat(sc.tq_khat(K, il, tqc[(hd, b)]), Q, refs)["kl"]
        # noiseshape codec (Hadamard+amax / mse) + SQuat, FULL-K
        Kt = torch.from_numpy(np.ascontiguousarray(K.transpose(1, 0, 2)))
        Xr = torch.stack([Kt[j] @ R.T for j in range(nkv)])
        can_shape = SHAPE_R <= hd // 2
        for bits in (2, 3):
            for sch in ("amax", "mse"):
                # plain (no subspace)
                plain = []
                for h in range(nkv):
                    if sch == "amax":
                        Yh = ns.quant_amax(Xr[h], hd, bits)
                    else:
                        Yh = ns.quant_mse(Xr[h], hd, bits)
                    plain.append((Yh @ R).numpy())
                rec[f"plain_b{bits}_{sch}"] = sc.eval_khat(
                    np.stack(plain, 1).astype(np.float32), Q, refs)["kl"]
                if not can_shape:
                    continue
                # SQuat calib (honest) + oracle (eval-fit)
                for srckey, qsl in (("cal", slice(0, Tc)), ("orc", slice(Tc, T))):
                    heads = []
                    for h in range(nkv):
                        qidx = np.where(qh_to_kv == h)[0]
                        Qsrc = (torch.from_numpy(np.ascontiguousarray(
                            Q[qsl, qidx, :].reshape(-1, hd))) @ R.T)
                        heads.append(squat_khat(Xr[h], Qsrc, R, hd, bits, sch))
                    rec[f"squat_{srckey}_b{bits}_{sch}"] = sc.eval_khat(
                        np.stack(heads, 1).astype(np.float32), Q, refs)["kl"]
        rows.append(rec)
        print(f"  L{il:2d} hd={hd} TQ3={rec['tq3']:.4f} "
              f"amax3={rec.get('plain_b3_amax', float('nan')):.4f} "
              f"mse3={rec.get('plain_b3_mse', float('nan')):.4f} "
              f"SQcal3={rec.get('squat_cal_b3_mse', float('nan')):.4f} "
              f"SQorc3={rec.get('squat_orc_b3_mse', float('nan')):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    out = SCRATCH / f"attack2_{short}.json"
    out.write_text(json.dumps(rows))

    def m(k):
        v = [r[k] for r in rows if k in r and not np.isnan(r[k])]
        return float(np.mean(v)) if v else float("nan")
    print(f"\n== {cfg['name']} AGGREGATE (mean over {len(rows)} layers) ==")
    print("--- 3-bit ladder (3.5 bpe) ---")
    print(f"  TQ3 (TurboQuant, study bar) : {m('tq3'):.4f}")
    print(f"  amax-3b (ns codec, no subsp): {m('plain_b3_amax'):.4f}")
    print(f"  mse-3b  (free scale,no subsp): {m('plain_b3_mse'):.4f}")
    print(f"  SQuat-3b calib amax         : {m('squat_cal_b3_amax'):.4f}")
    print(f"  SQuat-3b calib mse (=study) : {m('squat_cal_b3_mse'):.4f}")
    print(f"  SQuat-3b ORACLE mse (peeks) : {m('squat_orc_b3_mse'):.4f}")
    print("--- 2-bit ladder (2.5 bpe) ---")
    print(f"  TQ2 (TurboQuant)            : {m('tq2'):.4f}")
    print(f"  amax-2b (ns codec)          : {m('plain_b2_amax'):.4f}")
    print(f"  mse-2b  (free scale)        : {m('plain_b2_mse'):.4f}")
    print(f"  SQuat-2b calib mse          : {m('squat_cal_b2_mse'):.4f}")
    print(f"  SQuat-2b ORACLE mse         : {m('squat_orc_b2_mse'):.4f}")
    # decomposition of SQuat-3b(mse) win vs the study's TQ3 bar
    tq3, amax3, mse3 = m('tq3'), m('plain_b3_amax'), m('plain_b3_mse')
    sq3, sqo3 = m('squat_cal_b3_mse'), m('squat_orc_b3_mse')
    print("\n--- decomposition of 'SQuat-3b beats TQ3' (mse chain) ---")
    print(f"  TQ3 bar {tq3:.4f} -> amax3 {amax3:.4f}  (codec: {tq3-amax3:+.4f})")
    print(f"  amax3 {amax3:.4f} -> mse3 {mse3:.4f}     (free mse_opt: {amax3-mse3:+.4f})")
    print(f"  mse3 {mse3:.4f} -> SQuat-cal {sq3:.4f}   (SUBSPACE: {mse3-sq3:+.4f})")
    print(f"  SQuat-cal {sq3:.4f} vs ORACLE {sqo3:.4f} (oracle infl: {sq3-sqo3:+.4f})")
    if tq3 > sq3:
        tot = tq3 - sq3
        print(f"  total TQ3->SQuat win {tot:.4f}; subspace share "
              f"{(mse3-sq3)/tot*100:.0f}%, free(codec+mse) share "
              f"{(tq3-mse3)/tot*100:.0f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True)
    ap.add_argument("--layers", default="")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",") if x != ""] if a.layers else None
    run(a.short, layers)
