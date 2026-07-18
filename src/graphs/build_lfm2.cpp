#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include <algorithm>

// LFM2 / LFM2.5 (LiquidAI) — hybrid short-conv + attention models.
//
// Per-layer flow (both mixer kinds):
//   cur = x + mixer(RMSNorm_attn_norm(x));  out = cur + FFN(RMSNorm_ffn_norm(cur))
// Mixer is a short convolution when hparams.is_recurrent(il) (head_count_kv[il] == 0),
// else standard GQA attention with per-head q/k RMS norms followed by NEOX RoPE.
//
// Short-conv mixer (HF Lfm2ShortConv, no activation anywhere):
//   BCx = in_proj(cur) split into three n_embd chunks B, C, x (in that order)
//   conv_out = causal_conv1d(B * x)   with a rolling state of L_cache-1 columns
//   out = out_proj(C * conv_out)
//
// The rolling conv state lives in the hybrid recurrent cache kv_self.s_l[il]
// (F32, one row of n_embd*(L_cache-1) per sequence slot), the same machinery
// qwen3next/qwen35 use for their delta-net state.

// one short-conv mixer block operating on a single state slot
ggml_tensor * llm_build_context::build_lfm2_shortconv(ggml_cgraph * gf, ggml_tensor * input,
        ggml_tensor * inp_s_seq, ggml_tensor * inp_out_ids,
        int64_t state_seq_id, bool reset_state, int il) {

    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.ssm_in     != nullptr);
    GGML_ASSERT(layer.ssm_conv1d != nullptr);
    GGML_ASSERT(layer.ssm_out    != nullptr);

    const int64_t d_conv    = hparams.n_shortconv_l_cache;
    const int64_t n_tok     = input->ne[1];
    const int64_t state_dim = (d_conv - 1) * n_embd;

    ggml_tensor * state_storage = kv_self.s_l[il];
    GGML_ASSERT(state_storage != nullptr);
    GGML_ASSERT(state_storage->type == GGML_TYPE_F32); // ggml_ssm_conv reads it in place
    GGML_ASSERT(state_storage->ne[0] >= state_dim);
    const uint32_t n_slots = (uint32_t) state_storage->ne[1];
    GGML_ASSERT(state_seq_id >= 0 && (uint32_t) state_seq_id < n_slots);

    // pre-mixer norm (HF operator_norm)
    ggml_tensor * cur = llm_build_norm(ctx0, input, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "attn_norm", il);

    // in_proj -> {3*n_embd, n_tok}; chunk order is B, C, x
    ggml_tensor * bcx = llm_build_lora_mm(lctx, ctx0, layer.ssm_in, cur);
    cb(bcx, "shortconv_in_proj", il);

    const size_t nb_chunk = n_embd * ggml_element_size(bcx);
    ggml_tensor * b = ggml_view_2d(ctx0, bcx, n_embd, n_tok, bcx->nb[1], 0*nb_chunk);
    ggml_tensor * c = ggml_view_2d(ctx0, bcx, n_embd, n_tok, bcx->nb[1], 1*nb_chunk);
    ggml_tensor * x = ggml_view_2d(ctx0, bcx, n_embd, n_tok, bcx->nb[1], 2*nb_chunk);

    ggml_tensor * bx = ggml_mul(ctx0, b, x); // {n_embd, n_tok}
    cb(bx, "shortconv_bx", il);

    // this sequence's state slot, {state_dim, 1}
    const size_t state_row = state_storage->nb[1];
    ggml_tensor * state_dst = ggml_view_2d(ctx0, state_storage, state_dim, 1, state_row, (size_t) state_seq_id * state_row);
    ggml_tensor * state = state_dst;
    if (reset_state) {
        state = ggml_scale(ctx0, state, 0.0f);
        cb(state, "shortconv_state_reset", il);
    }
    ggml_tensor * conv_states = ggml_reshape_3d(ctx0, state, d_conv - 1, n_embd, 1);
    cb(conv_states, "shortconv_state", il);

    // rolling causal conv over [state | bx]
    // output layout: {n_embd*n_tok conv outputs | d_conv*n_embd*1 final windows}
    ggml_tensor * x_conv = ggml_ssm_conv(ctx0, conv_states, bx, layer.ssm_conv1d, inp_s_seq, nullptr);
    cb(x_conv, "shortconv_conv_raw", il);

    ggml_tensor * conv_out = ggml_view_2d(ctx0, x_conv, n_embd, n_tok, n_embd*ggml_element_size(x_conv), 0);

    // persist the new conv state: last d_conv-1 columns of the final per-channel window
    ggml_tensor * new_state = ggml_view_2d(ctx0, x_conv, d_conv - 1, n_embd,
            d_conv*ggml_element_size(x_conv), (1 + n_embd*n_tok)*ggml_element_size(x_conv));
    ggml_tensor * new_state_flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, new_state), state_dim, 1);
    ggml_tensor * state_cpy = ggml_cpy(ctx0, new_state_flat, state_dst);
    cb(state_cpy, "shortconv_state_cpy", il);
    ggml_build_forward_expand(gf, state_cpy);

    // gate with C, then out-project
    ggml_tensor * y = ggml_mul(ctx0, conv_out, c);
    cb(y, "shortconv_y", il);

    cur = llm_build_lora_mm(lctx, ctx0, layer.ssm_out, y);
    cb(cur, "shortconv_out", il);

    // residual
    if (inp_out_ids) {
        cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
        input = ggml_get_rows(ctx0, input, inp_out_ids);
    }
    cur = ggml_add(ctx0, cur, input);
    cb(cur, "shortconv_block_out", il);

    return cur;
}

ggml_cgraph * llm_build_context::build_lfm2() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inpL    = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    // per-token state-slot selector for ggml_ssm_conv (always zeros: the conv state
    // view passed to the op holds a single local slot, same contract as qwen3next)
    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    const float KQ_scale = 1.0f/sqrtf(float(n_embd_head));

    // per-token sequence bookkeeping (same contract as delta_net for qwen3next):
    // the decode loop chunks hybrid-arch batches so repeated seq ids never mix
    std::vector<llama_seq_id> token_seq_ids(std::max<int64_t>(n_tokens, 1), 0);
    if (batch.n_seq_id != nullptr && batch.seq_id != nullptr) {
        for (int64_t i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(batch.n_seq_id[i] > 0 && "LFM2 expects each token to belong to at least one sequence");
            GGML_ASSERT(batch.n_seq_id[i] == 1 && "LFM2 does not support multi-sequence tokens");
            token_seq_ids[i] = batch.seq_id[i][0];
        }
    }
    const llama_seq_id seq0 = token_seq_ids[0];
    const bool all_same_seq = std::all_of(token_seq_ids.begin(), token_seq_ids.begin() + n_tokens,
            [seq0](llama_seq_id s) { return s == seq0; });

    ggml_tensor * cur = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inp_out_l = il == n_layer - 1 ? inp_out_ids : nullptr;

        if (hparams.is_recurrent(il)) {
            if (all_same_seq) {
                const bool reset_state = batch.pos != nullptr && batch.pos[0] == 0;
                cur = build_lfm2_shortconv(gf, inpL, lctx.inp_s_seq_qnext, inp_out_l, seq0, reset_state, il);
            } else {
                // mixed-sequence batch: run each token against its own state slot
                ggml_tensor * out = nullptr;
                for (int64_t i = 0; i < n_tokens; ++i) {
                    ggml_tensor * in_i = ggml_view_2d(ctx0, inpL, inpL->ne[0], 1, inpL->nb[1], (size_t) i * inpL->nb[1]);
                    ggml_tensor * sq_i = ggml_view_2d(ctx0, lctx.inp_s_seq_qnext, 1, 1,
                            lctx.inp_s_seq_qnext->nb[1], (size_t) i * lctx.inp_s_seq_qnext->nb[1]);
                    const bool reset_i = batch.pos != nullptr && batch.pos[i] == 0;
                    ggml_tensor * out_i = build_lfm2_shortconv(gf, in_i, sq_i, nullptr, token_seq_ids[i], reset_i, il);
                    out = out == nullptr ? out_i : ggml_concat(ctx0, out, out_i, 1);
                }
                if (inp_out_l) {
                    out = ggml_get_rows(ctx0, out, inp_out_l);
                }
                cur = out;
            }
        } else {
            cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, inp_out_l, nullptr,
                    KQ_mask, nullptr, nullptr, KQ_scale, 0.0f, 0, il, true, false, true);
        }

        if (model.arch == LLM_ARCH_LFM2MOE && (uint32_t) il >= hparams.n_layer_dense_lead) {
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_gate_inp,  nullptr,
                    model.layers[il].ffn_up_exps,   nullptr,
                    model.layers[il].ffn_gate_exps, nullptr,
                    model.layers[il].ffn_down_exps, nullptr,
                    model.layers[il].ffn_exp_probs_b,
                    nullptr, nullptr, // no shared expert
                    nullptr, nullptr,
                    nullptr, nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true /*norm_w: norm_topk_prob*/, false, 0.0f,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SILU, cb, il, gf, true /*add_input*/, model.layers[il].ffn_up_gate_exps);
        } else {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true, false);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    // final norm ("token_embd_norm" in the GGUF) + lm head
    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
