#!/usr/bin/env python3
"""Full 34-layer gemma3 truncation sweep (the BEST faithful low-rank config, per
finding-1 verification) to find the exact bpe where eOptShrink-family reaches
full-K TQ3 quality under study / paper / per-vector accounting."""
import sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0,"/home/junc/Lean_llama.cpp/kit-v2")
import contour_study as cs
import eoptshrink_study as es
import torch
torch.set_num_threads(6)

cfg=[c for c in cs.MODELS if c["name"]=="gemma3-4b"][0]
Ks=cs.read_kcal_layers(es.ROOT/cfg["kf"]); Qs=cs.read_kcal_layers(es.ROOT/cfg["qf"])
layers=sorted(Ks)
max_il=max(layers)+1
tq={}
for il in layers:
    hd=Ks[il].shape[2]
    for b in (2,3):
        if (hd,b) not in tq:
            tq[(hd,b)]=cs.TurboQuantizer(n_layers=max_il,head_dim=hd,bits=b,group_size=None,
                rotation_strategy="randomized_hadamard",use_qjl=False,seed=42,device="cpu")

RANKS=[16,20,24,28]
kl_trunc={r:[] for r in RANKS}; kl_shrink={r:[] for r in RANKS}; kl_tq3=[]
Ts=[]; hds=[]
for il in layers:
    K,Q=Ks[il],Qs[il]; T,nkv,hd=K.shape; Tc=T//2
    Ts.append(T); hds.append(hd)
    refs=es.build_layer(K,Q,il,cfg); svd_cache=es.svd_calib(K,Tc)
    kl_tq3.append(es.eval_khat(es.tq_khat(K,il,tq[(hd,3)]),Q,refs)["kl"])
    for r in RANKS:
        kh_t,_=es.eopt_khat(K,il,r,"trunc",2,4,tq[(hd,2)],svd_cache)
        kl_trunc[r].append(es.eval_khat(kh_t,Q,refs)["kl"])
        kh_s,_=es.eopt_khat(K,il,r,"opt",2,4,tq[(hd,2)],svd_cache)
        kl_shrink[r].append(es.eval_khat(kh_s,Q,refs)["kl"])
    print(f"  layer {il} done", flush=True)

T=float(np.mean(Ts)); hd=float(np.mean(hds)); tq3=float(np.mean(kl_tq3))
def bpe_study(r): return 16*r/T + 4*r/hd + 16/hd + 2 + 16/32
def bpe_paper(r): return 2 + r*(128+hd)*4/(128*hd) + 16/hd + 16*r/(128*hd)
def bpe_pervec(r): return 16*r/T + 4*r/hd + 16*r/(T*hd) + 2 + 32/hd
tq3_bpe=dict(study=3+16/32, paper=3+16/hd, pervec=3+32/hd)
print(f"\ngemma3 full 34-layer: TQ3 KL={tq3:.4f}  bpe study={tq3_bpe['study']:.3f} paper={tq3_bpe['paper']:.3f} pervec={tq3_bpe['pervec']:.3f}")
print(f"{'r':>3s} | {'trunc KL':>9s} {'shrink KL':>9s} | {'bStudy':>7s} {'bPaper':>7s} {'bPervec':>7s} | trunc KL/TQ3")
for r in RANKS:
    kt=float(np.mean(kl_trunc[r])); ks=float(np.mean(kl_shrink[r]))
    print(f"{r:3d} | {kt:9.4f} {ks:9.4f} | {bpe_study(r):7.3f} {bpe_paper(r):7.3f} {bpe_pervec(r):7.3f} | {kt/tq3:5.2f}x")
# crossover: interpolate the rank where trunc KL == tq3, report its pervec/paper bpe
kts=np.array([np.mean(kl_trunc[r]) for r in RANKS])
if kts.min()<=tq3<=kts.max():
    r_cross=float(np.interp(-tq3,-kts[::-1],np.array(RANKS,float)[::-1]))  # kt decreasing in r
    print(f"\n-> truncation reaches TQ3 quality at r≈{r_cross:.1f}: "
          f"study={bpe_study(r_cross):.2f} paper={bpe_paper(r_cross):.2f} pervec={bpe_pervec(r_cross):.2f} bpe")
    print(f"   TQ3 sits at: study={tq3_bpe['study']:.2f} paper={tq3_bpe['paper']:.2f} pervec={tq3_bpe['pervec']:.2f}")
    for sc in ("study","paper","pervec"):
        b=dict(study=bpe_study,paper=bpe_paper,pervec=bpe_pervec)[sc](r_cross)
        print(f"   [{sc:6s}] eOptTrunc {b:.2f} vs TQ3 {tq3_bpe[sc]:.2f}  -> Δ={b-tq3_bpe[sc]:+.2f}  below 3.0? {b<3.0}")
