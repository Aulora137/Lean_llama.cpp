#!/usr/bin/env python3
"""Independently verify the OptShrink estimator (study finding #1).
Reuses the study's EXACT harness (import, read-only) but injects an independent
Gavish-Donoho Frobenius-optimal shrinker, and directly measures:
  (a) study gains vs GD gains, per component
  (b) final softmax-KL: study-shrink vs truncation vs GD-shrink vs no-lowrank
  (c) low-rank-ONLY reconstruction error (the paper's actual denoising claim):
      does shrinkage beat truncation when NO residual is stored?
"""
import sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/junc/Lean_llama.cpp/kit-v2")
import contour_study as cs
import eoptshrink_study as es
import torch
torch.set_num_threads(6)

cfg = [c for c in cs.MODELS if c["name"]=="gemma3-4b"][0]
Ks = cs.read_kcal_layers(es.ROOT/cfg["kf"]); Qs = cs.read_kcal_layers(es.ROOT/cfg["qf"])
layers = sorted(Ks)
sel = layers[2:len(layers):6]   # ~6 spread-out layers
print("layers:", sel)

# tq quantizers (match the study)
max_il=max(layers)+1
tq={}
for il in layers:
    hd=Ks[il].shape[2]
    for b in (2,3,4):
        if (hd,b) not in tq:
            tq[(hd,b)]=cs.TurboQuantizer(n_layers=max_il, head_dim=hd, bits=b,
                group_size=None, rotation_strategy="randomized_hadamard",
                use_qjl=False, seed=42, device="cpu")

def gd_gains(sv, n, m, r):
    """Gavish-Donoho Frobenius-optimal shrinker, median noise estimate. Independent of study."""
    beta = min(n,m)/max(n,m)
    mpmed = es._mp_median(beta)                       # median of MP in eigenvalue scale
    s_med = float(np.median(sv))
    sigma = s_med/(math.sqrt(max(n,m))*math.sqrt(mpmed))  # GD median noise estimate
    y = sv[:r]/(sigma*math.sqrt(max(n,m)))            # normalized singular values
    edge = 1+math.sqrt(beta)
    g=np.ones(r)
    for i in range(r):
        yi=y[i]
        if yi<=edge:
            g[i]=0.0; continue
        etaF=math.sqrt(max((yi*yi-beta-1)**2-4*beta,0.0))/yi
        g[i]=min(max(etaF/yi,0.0),1.0)
    return g

def lowrank_recon(K, h, r, gains):
    """low-rank-only recon [T,hd] using calib-fit V and given per-comp gains."""
    Tc=K.shape[0]//2
    M=torch.from_numpy(np.ascontiguousarray(K[:Tc,h,:])).float()
    U,sv,Vh=torch.linalg.svd(M, full_matrices=False)
    V=Vh[:r].T
    Kt=torch.from_numpy(np.ascontiguousarray(K[:,h,:])).float()
    c=(Kt@V)*torch.from_numpy(gains).float()[None,:]
    return (c@V.T).numpy(), sv.numpy()

allstats=dict(g_study=[], g_gd=[])
lr_err_shrink=[]; lr_err_trunc=[]; lr_err_gd=[]
kl_shrink=[]; kl_trunc=[]; kl_gd=[]; kl_nolr=[]; kl_tq2=[]; kl_tq3=[]
R=16
for il in sel:
    K,Q=Ks[il],Qs[il]; T,nkv,hd=K.shape; Tc=T//2
    n,m=Tc,hd
    refs=es.build_layer(K,Q,il,cfg)
    svd_cache=es.svd_calib(K,Tc)
    # gains comparison per head
    for h in range(nkv):
        sv,Vh=svd_cache[h]
        gs=es.optshrink_gains(sv,n,m,R)
        gg=gd_gains(sv,n,m,R)
        allstats["g_study"].append(gs); allstats["g_gd"].append(gg)
        # low-rank-only recon error (Frobenius, full K) for shrink/trunc/gd
        Kt=K[:,h,:].astype(np.float32)
        for tag,gains,store in (("shrink",gs,lr_err_shrink),("trunc",np.ones(R),lr_err_trunc),("gd",gg,lr_err_gd)):
            lr,_=lowrank_recon(K,h,R,gains)
            store.append(np.linalg.norm(lr-Kt)/np.linalg.norm(Kt))
    # final softmax-KL via study harness: shrink / trunc / gd / no-lowrank
    # study shrink (gain='opt'), trunc (gain='trunc'), both r=16 b=2
    kh_s,_=es.eopt_khat(K,il,R,"opt",2,4,tq[(hd,2)],svd_cache)
    kh_t,_=es.eopt_khat(K,il,R,"trunc",2,4,tq[(hd,2)],svd_cache)
    kl_shrink.append(es.eval_khat(kh_s,Q,refs)["kl"])
    kl_trunc.append(es.eval_khat(kh_t,Q,refs)["kl"])
    # GD shrink: replicate eopt_khat with injected gains
    lowrank=np.zeros((T,nkv,hd),dtype=np.float32)
    for h in range(nkv):
        sv,Vh=svd_cache[h]; V=torch.from_numpy(Vh[:R].T).float()
        gg=torch.from_numpy(gd_gains(sv,n,m,R)).float()
        Kt=torch.from_numpy(np.ascontiguousarray(K[:,h,:])).float()
        c=es.quant_uniform_pertoken((Kt@V)*gg[None,:],4)
        lowrank[:,h,:]=(c@V.T).numpy()
    residual=K.astype(np.float32)-lowrank
    xr=torch.from_numpy(np.ascontiguousarray(residual.transpose(1,0,2))).float()[None]
    with torch.no_grad():
        qkv=tq[(hd,2)].quantize(xr,layer_idx=il); rh=tq[(hd,2)].dequantize(qkv,layer_idx=il,apply_inverse_rot=True)
    kh_gd=lowrank+rh.squeeze(0).numpy().transpose(1,0,2).astype(np.float32)
    kl_gd.append(es.eval_khat(kh_gd,Q,refs)["kl"])
    # baselines
    kl_tq2.append(es.eval_khat(es.tq_khat(K,il,tq[(hd,2)]),Q,refs)["kl"])
    kl_tq3.append(es.eval_khat(es.tq_khat(K,il,tq[(hd,3)]),Q,refs)["kl"])
    print(f"  layer {il}: shrink {kl_shrink[-1]:.4f} trunc {kl_trunc[-1]:.4f} gd {kl_gd[-1]:.4f} | tq2 {kl_tq2[-1]:.4f} tq3 {kl_tq3[-1]:.4f}", flush=True)

gS=np.concatenate(allstats["g_study"]); gG=np.concatenate(allstats["g_gd"])
print("\n=== GAINS (r=16, all heads/layers) ===")
print(f"  study gains: mean {gS.mean():.3f}  min {gS.min():.3f}  max {gS.max():.3f}  frac<0.99 {(gS<0.99).mean():.2f}")
print(f"  GD    gains: mean {gG.mean():.3f}  min {gG.min():.3f}  max {gG.max():.3f}  frac<0.99 {(gG<0.99).mean():.2f}")
print(f"  mean|study-GD| per component: {np.abs(gS-gG).mean():.3f}")
print("\n=== LOW-RANK-ONLY recon rel-err (the paper's denoising claim: shrink should BEAT trunc) ===")
print(f"  shrink {np.mean(lr_err_shrink):.4f}   trunc {np.mean(lr_err_trunc):.4f}   gd {np.mean(lr_err_gd):.4f}")
print(f"  -> shrink beats trunc for low-rank-only? {np.mean(lr_err_shrink)<np.mean(lr_err_trunc)}")
print("\n=== FINAL softmax-KL (low-rank + 2-bit TQ residual) ===")
print(f"  study-shrink {np.mean(kl_shrink):.4f}   trunc {np.mean(kl_trunc):.4f}   GD-shrink {np.mean(kl_gd):.4f}")
print(f"  tq2 {np.mean(kl_tq2):.4f}   tq3 {np.mean(kl_tq3):.4f}")
print(f"  -> trunc beats study-shrink? {np.mean(kl_trunc)<np.mean(kl_shrink)}  by {100*(np.mean(kl_shrink)-np.mean(kl_trunc))/np.mean(kl_shrink):+.1f}%")
print(f"  -> GD-shrink beats trunc?    {np.mean(kl_gd)<np.mean(kl_trunc)}  (GD vs trunc {100*(np.mean(kl_gd)-np.mean(kl_trunc))/np.mean(kl_trunc):+.1f}%)")
