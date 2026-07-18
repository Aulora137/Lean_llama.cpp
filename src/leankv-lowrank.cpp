// LeanKV Phase A: low-rank K projection — loader + constant materialization.
// See leankv-lowrank.h for the file format and env-var documentation.

#include "leankv-lowrank.h"

#include "llama-model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

constexpr uint32_t LOWRANK_FILE_MAGIC   = 0x524C4B4Cu; // 'LKLR'
constexpr uint32_t LOWRANK_FILE_VERSION = 1u;
constexpr uint32_t LOWRANK_MAX_ENTRIES  = 4096u;
constexpr uint32_t LOWRANK_MAX_DIM      = 16384u;

// One materialized layer: tensor metadata lives in `ctx`, storage in `buf`.
// Both stay alive for the process lifetime (same policy as the leankv-calib
// runtime LUT globals) so graphs built at any later point may reference the
// tensors.
struct lowrank_layer_storage {
    leankv_lowrank_entry   entry;
    struct ggml_context  * ctx = nullptr;
    ggml_backend_buffer_t  buf = nullptr;
};

struct lowrank_state {
    bool initialized = false;
    bool active      = false; // fast gate for leankv_lowrank_get
    std::unordered_map<int, lowrank_layer_storage> layers;
};

lowrank_state g_lowrank;

// Raw file entry, host side.
struct lowrank_file_entry {
    uint32_t layer_idx = 0;
    uint32_t head_dim  = 0;
    uint32_t rank      = 0;
    std::vector<float> p; // row-major [head_dim][rank]: p[d*rank + j] = P[d][j]
};

bool lowrank_read_file(const char * path, std::vector<lowrank_file_entry> & out) {
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "leankv-lowrank: ERROR: cannot open '%s'\n", path);
        return false;
    }

    auto fail = [&](const char * msg) {
        std::fprintf(stderr, "leankv-lowrank: ERROR: %s in '%s'\n", msg, path);
        std::fclose(f);
        return false;
    };

    uint32_t hdr[3];
    if (std::fread(hdr, sizeof(uint32_t), 3, f) != 3) return fail("truncated header");
    if (hdr[0] != LOWRANK_FILE_MAGIC)   return fail("bad magic");
    if (hdr[1] != LOWRANK_FILE_VERSION) return fail("unsupported version");
    const uint32_t n_entries = hdr[2];
    if (n_entries == 0 || n_entries > LOWRANK_MAX_ENTRIES) return fail("implausible n_entries");

    out.clear();
    out.reserve(n_entries);
    for (uint32_t i = 0; i < n_entries; i++) {
        uint32_t ehdr[3];
        if (std::fread(ehdr, sizeof(uint32_t), 3, f) != 3) return fail("truncated entry header");
        lowrank_file_entry e;
        e.layer_idx = ehdr[0];
        e.head_dim  = ehdr[1];
        e.rank      = ehdr[2];
        if (e.head_dim == 0 || e.head_dim > LOWRANK_MAX_DIM ||
            e.rank == 0 || e.rank > e.head_dim) {
            return fail("implausible entry dims");
        }
        const size_t n = (size_t) e.head_dim * e.rank;
        e.p.resize(n);
        if (std::fread(e.p.data(), sizeof(float), n, f) != n) return fail("truncated entry data");
        out.push_back(std::move(e));
    }
    std::fclose(f);
    return true;
}

// Materialize the p_down/p_up pair for one file entry as backend-resident
// constants. Follows the llm_prepare_* "computed constant" pattern in
// llama.cpp: metadata in a private no_alloc ggml_context, storage allocated
// via ggml_backend_alloc_ctx_tensors_from_buft, data uploaded once with
// ggml_backend_tensor_set, buffer flagged as WEIGHTS.
bool lowrank_materialize(const lowrank_file_entry & fe,
                         ggml_backend_buffer_type_t buft,
                         lowrank_layer_storage    & out) {
    const int64_t head_dim = fe.head_dim;
    const int64_t rank     = fe.rank;

    ggml_init_params ip {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    // ggml_mul_mat(A, B) contracts ne[0] of both args, so:
    //   p_down ne=[head_dim, rank]: mul_mat(p_down, k[head_dim,...]) = P^T k
    //   p_up   ne=[rank, head_dim]: mul_mat(p_up,   c[rank,...])     = P c
    ggml_tensor * p_down = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, head_dim, rank);
    ggml_tensor * p_up   = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, rank, head_dim);
    ggml_format_name(p_down, "leankv_lowrank_down-%d", (int) fe.layer_idx);
    ggml_format_name(p_up,   "leankv_lowrank_up-%d",   (int) fe.layer_idx);

    out.buf = ggml_backend_alloc_ctx_tensors_from_buft(out.ctx, buft);
    if (!out.buf) {
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_set_usage(out.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // File layout is row-major [head_dim][rank]: fe.p[d*rank + j] = P[d][j].
    //   p_up flat [i1=d][i0=j] -> offset d*rank + j  == file layout, copy through.
    //   p_down flat [i1=j][i0=d] -> offset j*head_dim + d == transposed file layout.
    std::vector<float> down_host((size_t) head_dim * rank);
    for (int64_t d = 0; d < head_dim; d++) {
        for (int64_t j = 0; j < rank; j++) {
            down_host[(size_t) j*head_dim + d] = fe.p[(size_t) d*rank + j];
        }
    }
    ggml_backend_tensor_set(p_up,   fe.p.data(),      0, ggml_nbytes(p_up));
    ggml_backend_tensor_set(p_down, down_host.data(), 0, ggml_nbytes(p_down));

    out.entry.p_down   = p_down;
    out.entry.p_up     = p_up;
    out.entry.rank     = (int) rank;
    out.entry.head_dim = (int) head_dim;
    return true;
}

} // namespace

void leankv_lowrank_init(const struct llama_model & model) {
    if (g_lowrank.initialized) return;
    g_lowrank.initialized = true;

    const char * path = std::getenv("LEANKV_KV_LOWRANK");
    if (!path || !path[0]) return; // hard no-op: no file access when unset

    std::vector<lowrank_file_entry> entries;
    if (!lowrank_read_file(path, entries)) return;

    const int n_layer = (int) model.hparams.n_layer;
    for (const auto & fe : entries) {
        const int il = (int) fe.layer_idx;
        if (il < 0 || il >= n_layer) {
            std::fprintf(stderr,
                "leankv-lowrank: WARNING: layer %d out of range (n_layer=%d) — skipping\n",
                il, n_layer);
            continue;
        }
        if (g_lowrank.layers.count(il)) {
            std::fprintf(stderr,
                "leankv-lowrank: WARNING: duplicate entry for layer %d — skipping\n", il);
            continue;
        }
        const int model_head_dim = (int) model.hparams.n_embd_head_k(il);
        if ((int) fe.head_dim != model_head_dim) {
            std::fprintf(stderr,
                "leankv-lowrank: WARNING: layer %d head_dim %u != model %d — skipping\n",
                il, fe.head_dim, model_head_dim);
            continue;
        }

        // Same device placement as the layer's K weight (mirrors how
        // llm_prepare_mla picks the buffer type for computed_wkv_b);
        // falls back to the CPU buffer type when unavailable.
        ggml_backend_buffer_type_t buft = nullptr;
        if (il < (int) model.layers.size() &&
            model.layers[il].wk && model.layers[il].wk->buffer) {
            buft = ggml_backend_buffer_get_type(model.layers[il].wk->buffer);
        }
        if (!buft) buft = ggml_backend_cpu_buffer_type();

        lowrank_layer_storage st;
        if (!lowrank_materialize(fe, buft, st)) {
            std::fprintf(stderr,
                "leankv-lowrank: WARNING: failed to materialize layer %d — skipping\n", il);
            continue;
        }
        std::fprintf(stderr, "leankv-lowrank: layer %d rank %d/%d\n",
                     il, st.entry.rank, st.entry.head_dim);
        g_lowrank.layers.emplace(il, std::move(st));
    }

    g_lowrank.active = !g_lowrank.layers.empty();
}

const leankv_lowrank_entry * leankv_lowrank_get(int il) {
    if (!g_lowrank.active) return nullptr;
    const auto it = g_lowrank.layers.find(il);
    return it == g_lowrank.layers.end() ? nullptr : &it->second.entry;
}
