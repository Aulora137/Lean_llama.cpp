#!/usr/bin/env python3
"""Verify HQMQ Med3x outlier rate (study finding #4) directly on real K.
Measures per-chunk (per-head median = study; global-pooled median = paper wording)
and per-element outlier rates, per arch. Also recomputes HQMQ honest bpe if
outliers were handled per-ELEMENT (fp16 value+index) instead of per-CHUNK."""
import sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0,"/home/junc/Lean_llama.cpp/kit-v2")
import contour_study as cs
import torch

ROOT=Path("/home/junc/Lean_llama.cpp")
MODELS={"gemma3-4b":("gemma3_kq_k.bin",), "gemma4-E2B":("e2b_kq_k.bin",), "LFM2.5-1.2B":("lfm2_kq_k.bin",)}
C=3.0

for short,(kf,) in MODELS.items():
    Ks=cs.read_kcal_layers(ROOT/kf)
    ils=sorted(Ks)
    perhead_rates=[]; global_rates=[]; elem_rates=[]; hds=[]
    # for per-element outlier accounting
    for il in ils:
        K=Ks[il]; T,nkv,hd=K.shape; nch=hd//4; hds.append(hd)
        Kt=torch.from_numpy(K).float()                       # [T,nkv,hd]
        ch=Kt.reshape(T,nkv,nch,4)
        radius=ch.norm(dim=-1)                               # [T,nkv,nch] per-chunk norm
        # (a) study: per-(layer,kv-head) median
        for h in range(nkv):
            med=radius[:,h,:].median().clamp_min(1e-12)
            perhead_rates.append(float((radius[:,h,:]>C*med).float().mean()))
        # (b) paper wording: global-pooled median across heads+tokens (per layer)
        gmed=radius.median().clamp_min(1e-12)
        global_rates.append(float((radius>C*gmed).float().mean()))
        # (c) per-element: |x| > C*median(|x|) pooled per (layer,kv-head)
        for h in range(nkv):
            ax=Kt[:,h,:].abs()
            emed=ax.median().clamp_min(1e-12)
            elem_rates.append(float((ax>C*emed).float().mean()))
    hd=int(np.mean(hds))
    p_head=float(np.mean(perhead_rates)); p_glob=float(np.mean(global_rates)); p_elem=float(np.mean(elem_rates))
    print(f"\n=== {short} (hd~{hd}) ===")
    print(f"  per-CHUNK Med3x, per-head median (STUDY): {p_head*100:5.2f}%")
    print(f"  per-CHUNK Med3x, global-pooled median   : {p_glob*100:5.2f}%")
    print(f"  per-ELEMENT Med3x (|x|>3*median|x|)      : {p_elem*100:5.2f}%")
    # HQMQ bpe for s24_r5 under per-chunk vs per-element outlier handling
    S=24; br=5; idx_bits=math.ceil(math.log2(24))+math.ceil(math.log2(S))
    per_chunk=(idx_bits+br)/4.0
    codebook=S*4*16/(731*hd)
    # study per-chunk: outlier chunk stored fp16 4-tuple (16 bpe) + 1-bit flag/chunk
    b_perchunk=(1-p_head)*per_chunk + p_head*16 + 1/4 + 16/hd + codebook
    # per-element alt: keep quantizing ALL chunks; add outlier elems as fp16 value(16)+index(log2 hd), rate p_elem of elements
    idx_cost=math.ceil(math.log2(hd))
    b_perelem = per_chunk + p_elem*(16+idx_cost) + 16/hd + codebook   # no chunk flag; per-elem side list
    print(f"  HQMQ s24_r5 honest bpe: per-CHUNK(study)={b_perchunk:.2f}   per-ELEMENT-alt={b_perelem:.2f}")
