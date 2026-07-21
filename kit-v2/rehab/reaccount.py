#!/usr/bin/env python3
"""Re-account eOptShrink & HQMQ bpe from the study's cached per-layer JSON under
several accounting schemes, and locate where each method reaches TQ3/TQ4 quality.
Uses ONLY the study's measured KL / r / p_out (no re-quantization)."""
import json, math
from pathlib import Path
import numpy as np

SCRATCH = Path("/tmp/claude-1000/-home-junc-rikuri-rikurinode/"
               "0a7b0d58-b4e2-4aa3-bb15-3e35bc56022f/scratchpad")
FILES = {"gemma3-4b": "eopt_gemma3-4b.json",
         "gemma4-E2B": "eopt_gemma4-E2B.json",
         "LFM2.5-1.2B": "eopt_LFM2.5-1.2B.json"}

EOPT_ARMS = ["eopt_r4_b2","eopt_r8_b2","eopt_r12_b2","eopt_r16_b2","eopt_r24_b2",
             "eopt_r32_b2","eopt_r16_trunc","eopt_r8_b3","eopt_r16_b3","eopt_auto_b2"]
# res_bits, coef_bits per arm
ARM_CFG = {
 "eopt_r4_b2":(2,4),"eopt_r8_b2":(2,4),"eopt_r12_b2":(2,4),"eopt_r16_b2":(2,4),
 "eopt_r24_b2":(2,4),"eopt_r32_b2":(2,4),"eopt_r16_trunc":(2,4),
 "eopt_r8_b3":(3,4),"eopt_r16_b3":(3,4),"eopt_auto_b2":(2,4),
}

def bpe_study(r,T,hd,rb,cb):        # study's eopt_bpe_honest
    return 16*r/T + cb*r/hd + 16/hd + rb + 16/32
def bpe_paper(r,T,hd,rb,cb,nb=128,bs=4):  # paper: b + r(n+d)bs/(nd) + 16/hd TQnorm + 16r/(nb*hd) singvals
    return rb + r*(nb+hd)*bs/(nb*hd) + 16/hd + 16*r/(nb*hd)
def bpe_pervec(r,T,hd,rb,cb):       # actual codec: per-vector 32/hd resid scale; fp16 basis over T; per-comp coef scale
    # residual per-vector scale = 32/hd (quantizer.py bit_accounting); basis fp16/T; coef 4-bit; coef scale per-component (r fp16 over T) ~16r/(T*hd) negligible
    return 16*r/T + cb*r/hd + 16*r/(T*hd) + rb + 32/hd
def bpe_tq_study(b,hd): return b + 16/32
def bpe_tq_paper(b,hd): return b + 16/hd
def bpe_tq_pervec(b,hd): return b + 32/hd

def agg(files):
    out={}
    for short,fn in files.items():
        L=json.load(open(SCRATCH/fn))["layers"]
        ils=sorted(L,key=int)
        rows={}
        # baselines
        for name in ("tq2","tq3","tq4"):
            rows[name]=dict(kl=np.mean([L[il][name]["kl"] for il in ils]),
                            hd=np.mean([L[il]["hd"] for il in ils]))
        for arm in EOPT_ARMS:
            if arm not in L[ils[0]]: continue
            kl=np.mean([L[il][arm]["kl"] for il in ils])
            rb,cb=ARM_CFG[arm]
            bs,bp,bv=[],[],[]
            for il in ils:
                e=L[il][arm]; hd=L[il]["hd"]; T=L[il]["T"]; r=e["r"]
                bs.append(bpe_study(r,T,hd,rb,cb)); bp.append(bpe_paper(r,T,hd,rb,cb)); bv.append(bpe_pervec(r,T,hd,rb,cb))
            rows[arm]=dict(kl=kl,r=np.mean([L[il][arm]["r"] for il in ils]),
                           bpe_study=np.mean(bs),bpe_paper=np.mean(bp),bpe_pervec=np.mean(bv))
        # hqmq
        for arm in ("hqmq_s24_r5","hqmq_s48_r5","hqmq_s96_r6"):
            if arm not in L[ils[0]]: continue
            rows[arm]=dict(kl=np.mean([L[il][arm]["kl"] for il in ils]),
                           p_out=np.mean([L[il][arm]["p_out"] for il in ils]),
                           S=L[ils[0]][arm]["S"])
        # tq baselines bpe under schemes
        hd=rows["tq2"]["hd"]
        rows["_tqbpe"]={b:dict(study=bpe_tq_study(b,hd),paper=bpe_tq_paper(b,hd),pervec=bpe_tq_pervec(b,hd)) for b in (2,3,4)}
        out[short]=rows
    return out

def main():
    A=agg(FILES)
    for short in FILES:
        R=A[short]; hd=R["tq2"]["hd"]
        tq2,tq3,tq4=R["tq2"]["kl"],R["tq3"]["kl"],R["tq4"]["kl"]
        tqb=R["_tqbpe"]
        print(f"\n================= {short}  (hd~{hd:.0f}) =================")
        print(f"  TQ2 KL={tq2:.4f}  TQ3 KL={tq3:.4f}  TQ4 KL={tq4:.4f}")
        print(f"  TQ3 bpe: study={tqb[3]['study']:.3f} paper={tqb[3]['paper']:.3f} pervec={tqb[3]['pervec']:.3f}")
        print(f"  {'arm':14s} {'KL':>7s} {'r':>5s} | {'bStudy':>7s} {'bPaper':>7s} {'bPervec':>7s} | KL/TQ3")
        for arm in EOPT_ARMS:
            if arm not in R: continue
            e=R[arm]
            print(f"  {arm:14s} {e['kl']:7.4f} {e['r']:5.1f} | {e['bpe_study']:7.3f} {e['bpe_paper']:7.3f} {e['bpe_pervec']:7.3f} | {e['kl']/tq3:5.2f}x")
        # decisive: lowest bpe (each scheme) at which an eopt arm reaches TQ3 KL
        for scheme,tqbpe in (("study",tqb[3]["study"]),("paper",tqb[3]["paper"]),("pervec",tqb[3]["pervec"])):
            key=f"bpe_{scheme}"
            reach=[(R[a][key],R[a]["kl"],a) for a in EOPT_ARMS if a in R and R[a]["kl"]<=tq3]
            best=min(reach) if reach else None
            if best:
                print(f"  [{scheme:6s}] TQ3@{tqbpe:.2f}: first eopt<=TQ3 = {best[2]} KL={best[1]:.4f} @ {best[0]:.2f}bpe  (Δ={best[0]-tqbpe:+.2f} vs TQ3)")
            else:
                print(f"  [{scheme:6s}] no eopt arm reaches TQ3 KL")
        # HQMQ vs TQ4 with corrected outlier accounting
        print("  --- HQMQ vs TQ4 ---")
        for arm in ("hqmq_s24_r5","hqmq_s48_r5","hqmq_s96_r6"):
            if arm not in R: continue
            e=R[arm]; S=e["S"]; p=e["p_out"]
            idx_bits=math.ceil(math.log2(24))+math.ceil(math.log2(S))
            br=5 if arm!="hqmq_s96_r6" else 6
            per_chunk=(idx_bits+br)/4.0
            # study honest: outlier chunk=16bpe
            base=per_chunk+16/hd+S*4*16/(731*hd)+1/4  # +flag +radscale +codebook
            b_study=(1-p)*per_chunk+p*16+1/4+16/hd+S*4*16/(731*hd)
            # corrected: per-ELEMENT outlier (only outlier elems as fp16+index), rate p_elem measured separately below
            print(f"  {arm:12s} KL={e['kl']:.4f} p_chunk={p*100:4.1f}%  bStudy={b_study:.2f}  TQ4 KL={tq4:.4f} beats_quality={e['kl']<=tq4}")
if __name__=="__main__":
    main()
