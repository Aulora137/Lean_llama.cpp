**Ternary KV-cache quantization could work, but naive ({-1,0,+1}) quantization is unlikely to rescue the Gemma 4 failure you’re seeing.** It may actually be worse than a well-designed four-level 2-bit quantizer unless you add per-channel scaling, outlier handling, and probably mixed precision for sensitive layers.

Your observation about the error cascading is important. A small key error changes:

[
QK^\top
]

which changes the softmax attention distribution. That changes the layer output, so the next layer produces different (Q), (K), and (V). Thus the next layer is quantizing activations already shifted from the reference trajectory. This can produce exactly the progressive collapse you described.

## Why plain ternary probably fails

A basic ternary representation would be:

[
\hat{k}_i=s\cdot t_i,\qquad t_i\in{-1,0,+1}
]

with thresholding such as:

[
t_i=
\begin{cases}
-1 & k_i<-\tau\
0 & |k_i|\le\tau\
+1 & k_i>\tau
\end{cases}
]

The problem is that all nonzero values in a group receive the same magnitude (s). Keys usually contain:

* channel-dependent scales;
* important medium-magnitude values;
* a small number of large outliers;
* direction information that must be preserved for (QK^\top).

Ternary quantization preserves sign and sparsity reasonably well, but discards most magnitude information. A conventional asymmetric 2-bit quantizer has four reconstruction levels, while ternary has only three. Therefore, ternary is not automatically more accurate merely because zero is represented exactly.

Also, ternary consumes two physical bits per element in a straightforward implementation even though its information content is only:

[
\log_2(3)\approx1.585\text{ bits}.
]

Unless you entropy-pack it, the memory cost is approximately the same as ordinary INT2 but with one fewer reconstruction level.

## The approach most likely to work

For Gemma 4, I would test an **asymmetric hybrid scheme**, not ternary for the entire KV cache:

[
\boxed{
K:\text{ per-channel scaled ternary or INT3}
\qquad
V:\text{ per-token INT2}
}
]

KIVI found this same fundamental asymmetry: keys benefit strongly from per-channel quantization, while values work better with per-token quantization. ([Proceedings of Machine Learning Research][1])

### 1. Quantize K per channel

For each layer, KV head, and channel (c):

[
\hat{K}*{t,h,c}=s*{h,c},T_{t,h,c}
]

where:

[
T_{t,h,c}\in{-1,0,+1}.
]

The scale should be channel-specific:

[
s_{h,c}
=======

\operatorname*{argmin}*{s}
\sum_t
\left(K*{t,h,c}-sT_{t,h,c}\right)^2.
]

For a fixed ternary assignment, the least-squares scale is:

[
s_{h,c}
=======

\frac{\sum_t K_{t,h,c}T_{t,h,c}}
{\sum_t T_{t,h,c}^2}.
]

That is much safer than one scale for an entire 512-dimensional head.

A practical grouping hierarchy would be:

```text
layer
 └── KV head
      └── channel or small channel group
           ├── scale
           └── ternary codes over tokens
```

For memory and kernel efficiency, groups of 8–32 channels may be a compromise, but true per-channel scaling should be the accuracy baseline.

### 2. Quantize K before RoPE when possible

KVQuant observed that keys have structured outlier channels before RoPE, while RoPE mixes pairs of channels and makes those outliers harder to quantize. Its experiments found that pre-RoPE key quantization improved accuracy compared with post-RoPE quantization. 

So the pipeline could be:

```text
K projection
    ↓
per-channel ternary quantization
    ↓
store ternary K + channel scales
    ↓
dequantize during attention
    ↓
apply RoPE
    ↓
QKᵀ
```

This requires a fused dequantization-plus-RoPE kernel to avoid losing the performance benefit.

For shared-KV layers in Gemma 4, special care is required because some layers reuse cached K/V instead of computing new projections. ([Hugging Face][2]) An error in a shared cache may consequently affect several consuming layers, making those cache-producing layers particularly sensitive.

### 3. Keep outliers outside the ternary representation

Do not force the largest values into (\pm s). Use a dense-plus-sparse representation:

[
K=\hat{K}*{\text{ternary}}+K*{\text{outlier}}.
]

For example:

* ternary-encode 98–99.5% of elements;
* retain the largest 0.5–2% in FP16, BF16, FP8, or INT8;
* add their contribution during the query-key dot product.

KVQuant found that separating only about 1% of outliers substantially improved low-bit accuracy. 

This is probably more useful than zeroing a large fraction of the ordinary key values.

## Attention-aware ternary optimization

Ordinary MSE is not the best objective for keys. You do not fundamentally care whether:

[
K\approx\hat K.
]

You care whether:

[
QK^\top\approx Q\hat K^\top.
]

Therefore, choose thresholds and scales using an attention-weighted objective:

[
\mathcal L_K
============

\mathbb E_Q
\left[
\left|
QK^\top-Q\hat K^\top
\right|_F^2
\right].
]

An even stronger calibration loss would include softmax:

[
\mathcal L_{\text{attn}}
========================

D_{\mathrm{KL}}
\left(
\operatorname{softmax}(QK^\top/\sqrt d)
;\middle|;
\operatorname{softmax}(Q\hat K^\top/\sqrt d)
\right).
]

This matters because a small logit error near a close competition between tokens can radically change attention routing, while a larger error on an irrelevant token may do virtually nothing.

## Ternary should probably not use fixed symmetric thresholds

A fixed rule like:

[
|k|<0.5s\Rightarrow0
]

will likely be too crude. Calibrate a threshold (\tau_{l,h,c}) for every layer/head/channel, or at least every small channel group.

A useful parameterization is:

[
T(k;\tau)=
\begin{cases}
-1 & k<-\tau\
0 & -\tau\le k\le\tau\
+1 & k>\tau
\end{cases}
]

and jointly search for (s) and (\tau) to minimize attention-logit error.

You could also use asymmetric ternary reconstruction:

[
\hat k\in{-s_-,0,+s_+},
]

which still has three codes but allows positive and negative tails to have different magnitudes. That may help when a channel is skewed.

## Why the head geometry matters—but not exactly as sparsity

Gemma 4 reportedly uses hybrid attention with different head dimensions: many sliding-window layers use `head_dim=256`, while global-attention layers can use `head_dim=512`. ([GitHub][3]) The Transformers configuration also distinguishes attention heads from KV heads; Gemma 4 uses GQA and may have far fewer KV heads than query heads. ([Hugging Face][4])

So I would separate these concepts:

* `num_attention_heads`: number of query heads;
* `num_key_value_heads`: actual cached K/V heads;
* `head_dim`: coordinates in each head;
* projection or sketch dimension used by TurboQuant.

The condition “number of heads is less than head dimension” does not by itself mean the K tensor is sparse. But it can make a projection/sketch-based estimator poorly conditioned if its effective sample or projection dimension is too small relative to the 512-dimensional key vector.

TurboQuant’s strongest published claim is essentially around three-bit compression rather than universally lossless two-bit K quantization. Google describes it as combining vector transformations and quantized projection/error-correction techniques. ([Google Research][5]) So a two-bit collapse on unusual 512-dimensional global heads would not be shocking.

## My recommended precision map

For an initial experiment:

| Component                        | Suggested format                    |
| -------------------------------- | ----------------------------------- |
| Sliding-window K, `head_dim=256` | Per-channel ternary + FP16 outliers |
| Sliding-window V                 | Per-token asymmetric INT2           |
| Global K, `head_dim=512`         | INT3 initially                      |
| Global V                         | INT2 or INT3                        |
| Cache-producing shared-KV layers | INT3/INT4 or FP8                    |
| First and last few layers        | INT3/INT4                           |
| Recent 64–256 tokens             | FP8/FP16 residual window            |
| Older tokens                     | Quantized                           |

That would tell you whether the failure is concentrated in:

* 512-dimensional global-attention layers;
* shared-cache producer layers;
* early layers;
* particular outlier channels;
* recent-token quantization.

Only after establishing a stable INT3 baseline would I replace selected K layers with ternary.

## A useful stability diagnostic

At every layer (l), record:

[
E_K^{(l)}
=========

\frac{|K^{(l)}-\hat K^{(l)}|_F}
{|K^{(l)}|_F},
]

but more importantly:

[
E_{\text{logit}}^{(l)}
======================

\frac{
|Q^{(l)}K^{(l)\top}-Q^{(l)}\hat K^{(l)\top}|_F
}{
|Q^{(l)}K^{(l)\top}|_F
},
]

and:

[
D_{\text{attn}}^{(l)}
=====================

D_{\mathrm{KL}}
\left(A^{(l)}*{\rm fp}|A^{(l)}*{\rm quant}\right).
]

Run two tests:

1. **Isolated error:** feed each layer the FP reference hidden state but quantize that layer’s cache.
2. **Free-running error:** let quantized outputs propagate normally.

The gap between those tests measures the cascade. It will also identify the first layer where attention routing—not merely reconstruction MSE—breaks.

## Bottom line

A **ternary K cache is technically viable**, especially with the exact-zero state, but it should be treated as an attention-aware vector quantizer rather than simple rounding.

The credible version is:

[
\boxed{
\text{pre-RoPE}
+
\text{per-channel scales}
+
\text{learned thresholds}
+
\text{outlier residuals}
+
\text{mixed precision by layer}
}
]

Plain per-token ternary K quantization is very likely to collapse. For Gemma 4’s 512-dimensional global heads and reused KV caches, I would expect **INT3 K / INT2 V** to be the practical floor initially, with ternary introduced only in low-sensitivity K channels or layers.

[1]: https://proceedings.mlr.press/v235/liu24bz.html "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
[2]: https://huggingface.co/blog/gemma4?utm_source=chatgpt.com "Gemma 4: Frontier multimodal intelligence on device"
[3]: https://github.com/Dao-AILab/flash-attention/issues/2427?utm_source=chatgpt.com "Support head_dim=512 for Gemma 4 global attention layers"
[4]: https://huggingface.co/docs/transformers/en/model_doc/gemma4?utm_source=chatgpt.com "Gemma4"
[5]: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/ "TurboQuant: Redefining AI efficiency with extreme compression"
