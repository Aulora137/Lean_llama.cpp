// LeanKV Phase 7: K-vector calibration dump + in-memory Lloyd-Max fit.
// See leankv-calib.h for format and env-var documentation.

#include "leankv-calib.h"
#include "leankv-calib-corpus.h"
#include "leankv-codebook.h"

extern "C" {
#include "ggml-tq-calib.h"
}

#include "llama.h"
#include "llama-model.h"
#include <random> // std::mt19937, pulled in via llama-sampling.h below
#include "llama-context.h"
#include "llama-arch.h"

#include "ggml.h"
#include "ggml-backend.h"
extern "C" {
#include "ggml-tq-runtime.h" // LeanKV 7a Stage 4a: install fitted LUTs
}

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

bool leankv_calib_enabled() {
    static int cached = -1;
    if (cached == -1) {
        const char * env = std::getenv("LEANKV_CALIBRATION_DUMP");
        cached = (env && env[0] && !(env[0] == '0' && env[1] == 0)) ? 1 : 0;
    }
    return cached != 0;
}

// Runtime flag flipped by leankv_autocalibrate around its warm-up pass.
// We keep it separate from the env-var gate so `LEANKV_CALIBRATION_DUMP=1`
// (file-output mode) remains independent of the in-memory auto-calib path.
static std::atomic<bool> g_leankv_runtime_capture{false};

bool leankv_calib_capture_required() {
    return leankv_calib_enabled() || g_leankv_runtime_capture.load(std::memory_order_relaxed);
}

void leankv_calib_set_runtime_capture(bool on) {
    g_leankv_runtime_capture.store(on, std::memory_order_relaxed);
}

leankv_calib_state * leankv_calib_init_in_memory() {
    auto * s = new leankv_calib_state();
    s->accumulate_in_memory = true;
    s->file = nullptr;
    s->path[0] = '\0';
    s->max_records = 0;
    return s;
}

leankv_calib_state * leankv_calib_init() {
    if (!leankv_calib_enabled()) {
        return nullptr;
    }

    auto * s = new leankv_calib_state();

    const char * env_path = std::getenv("LEANKV_CALIBRATION_DUMP_PATH");
    const char * path = (env_path && env_path[0]) ? env_path : "leankv_k_calib.bin";
    std::snprintf(s->path, sizeof(s->path), "%s", path);

    s->file = std::fopen(s->path, "wb");
    if (!s->file) {
        std::fprintf(stderr, "leankv-calib: ERROR: failed to open '%s' for write\n", s->path);
        delete s;
        return nullptr;
    }

    const char * env_max = std::getenv("LEANKV_CALIBRATION_DUMP_MAX");
    if (env_max && env_max[0]) {
        s->max_records = std::atoi(env_max);
    }

    // File header
    const uint32_t magic   = 0x4C41434Bu; // 'KCAL'
    const uint32_t version = 1u;
    std::fwrite(&magic,   sizeof(uint32_t), 1, s->file);
    std::fwrite(&version, sizeof(uint32_t), 1, s->file);
    std::fflush(s->file);

    std::fprintf(stderr,
        "leankv-calib: dumping K-vectors to '%s' (max_records=%d)\n",
        s->path, s->max_records);

    return s;
}

void leankv_calib_free(leankv_calib_state * s) {
    if (!s) return;
    if (s->file) {
        std::fflush(s->file);
        std::fclose(s->file);
        std::fprintf(stderr,
            "leankv-calib: wrote %d record(s) to '%s'\n",
            s->n_records, s->path);
    }
    delete s;
}

// Decode a raw tensor byte buffer into a float vector for the observable
// elements (head_dim * n_rows). Returns 0 on unknown dtype.
static size_t decode_tensor_floats(const struct ggml_tensor * t,
                                   const uint8_t *            raw,
                                   size_t /*n_bytes*/,
                                   std::vector<float> &       out) {
    const size_t head_dim = (size_t) t->ne[0];
    size_t n_rows = (size_t) t->ne[1];
    if (ggml_n_dims(t) >= 3) n_rows *= (size_t) t->ne[2];
    if (ggml_n_dims(t) >= 4) n_rows *= (size_t) t->ne[3];
    const size_t n_elem = head_dim * n_rows;

    const size_t old_size = out.size();
    out.resize(old_size + n_elem);

    if (t->type == GGML_TYPE_F32) {
        const float * src = reinterpret_cast<const float *>(raw);
        std::memcpy(out.data() + old_size, src, n_elem * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(raw);
        for (size_t i = 0; i < n_elem; i++) {
            out[old_size + i] = ggml_fp16_to_fp32(src[i]);
        }
    } else {
        // Unsupported dtype — roll back.
        out.resize(old_size);
        return 0;
    }
    return n_elem;
}

bool leankv_calib_dump_tensor(leankv_calib_state * s, const struct ggml_tensor * t, int il) {
    if (!s || !t) return false;
    if (s->max_records > 0 && s->n_records >= s->max_records) return false;

    const size_t n_bytes = ggml_nbytes(t);
    if (n_bytes == 0) return false;

    // Pull the tensor data once via the backend — shared between the file
    // writer (dump mode) and the in-memory accumulator (first-load mode).
    std::vector<uint8_t> buf(n_bytes);
    ggml_backend_tensor_get(t, buf.data(), 0, n_bytes);

    if (s->file) {
        const uint32_t rec_magic = 0x52434B4Cu; // 'LKCR'
        const uint32_t il_u      = (uint32_t) il;
        const uint32_t dtype     = (uint32_t) t->type;
        const uint32_t ndims     = (uint32_t) ggml_n_dims(t);

        uint32_t ne[4] = {
            (uint32_t) t->ne[0], (uint32_t) t->ne[1],
            (uint32_t) t->ne[2], (uint32_t) t->ne[3],
        };
        uint32_t nb[4] = {
            (uint32_t) t->nb[0], (uint32_t) t->nb[1],
            (uint32_t) t->nb[2], (uint32_t) t->nb[3],
        };
        const uint32_t nbytes_u = (uint32_t) n_bytes;

        std::fwrite(&rec_magic, sizeof(uint32_t), 1, s->file);
        std::fwrite(&il_u,      sizeof(uint32_t), 1, s->file);
        std::fwrite(&dtype,     sizeof(uint32_t), 1, s->file);
        std::fwrite(&ndims,     sizeof(uint32_t), 1, s->file);
        std::fwrite(ne,         sizeof(uint32_t), 4, s->file);
        std::fwrite(nb,         sizeof(uint32_t), 4, s->file);
        std::fwrite(&nbytes_u,  sizeof(uint32_t), 1, s->file);
        std::fwrite(buf.data(), 1, n_bytes, s->file);
        std::fflush(s->file);
    }

    if (s->accumulate_in_memory) {
        size_t before = s->blocks.size();
        size_t added  = decode_tensor_floats(t, buf.data(), n_bytes, s->blocks);
        if (added > 0) {
            if (il + 1 > (int) s->layer_counts.size()) {
                s->layer_counts.resize(il + 1, 0);
            }
            s->layer_counts[il] += added;
            if (il > s->max_layer_seen) s->max_layer_seen = il;
            (void) before;
        }
    }

    s->n_records++;
    return true;
}

// ── Empirical codebook fitting (Phase 7a) ─────────────────────────

bool leankv_calib_fit_codebook(const leankv_calib_state * s,
                               int                        bits,
                               int                        block_size,
                               const char *               arch,
                               int                        n_layers,
                               leankv::codebook *         out) {
    if (!s || !out) return false;
    if (bits != 2 && bits != 3 && bits != 4) return false;
    if (block_size <= 0) return false;

    const int n_levels = 1 << bits;

    *out = leankv::codebook{};
    out->version  = leankv::CODEBOOK_FILE_VERSION;
    out->bits     = bits;
    out->n_levels = n_levels;
    out->n_layers = n_layers;
    if (arch) std::snprintf(out->arch, sizeof(out->arch), "%s", arch);

    const size_t total = s->blocks.size();
    const size_t n_full_elems = (total / (size_t) block_size) * (size_t) block_size;
    const size_t n_blocks     = n_full_elems / (size_t) block_size;

    // Need at least 4x the codebook size in samples to fit meaningfully.
    const size_t min_samples = (size_t)(n_levels * 16);
    if (n_blocks == 0 || n_full_elems < min_samples) {
        std::fprintf(stderr,
            "leankv-calib: too few samples (%zu, need >= %zu) — writing default marker\n",
            n_full_elems, min_samples);
        return false;
    }

    int iters = leankv_fit_empirical_codebook(
        s->blocks.data(), n_blocks, block_size, n_levels, 40, out->global_levels);
    (void) iters;

    if (!leankv::codebook_levels_ok(out->global_levels, n_levels)) {
        std::fprintf(stderr,
            "leankv-calib: global fit failed sanity check — writing default marker\n");
        return false;
    }

    // Compute SNR delta vs standard Gaussian levels so we can decide
    // whether the empirical fit is worth keeping.
    static const float gaussian_tq2[4] = {
        -1.0f, -0.2997714f, +0.2997714f, +1.0f,
    };
    static const float gaussian_tq3[8] = {
        -1.0f, -0.6245203f, -0.3513239f, -0.1138989f,
        +0.1138989f, +0.3513239f, +0.6245203f, +1.0f,
    };
    static const float gaussian_tq4[16] = {
        -1.0000000f, -0.7573038f, -0.5923403f, -0.4599576f,
        -0.3450764f, -0.2405254f, -0.1421261f, -0.0470277f,
        +0.0470277f, +0.1421261f, +0.2405254f, +0.3450764f,
        +0.4599576f, +0.5923403f, +0.7573038f, +1.0000000f,
    };
    const float * gaussian = bits == 2 ? gaussian_tq2
                           : bits == 3 ? gaussian_tq3
                                       : gaussian_tq4;

    // Block-normalize once and score both codebooks on that.
    std::vector<float> norm(n_full_elems);
    leankv_block_normalize(s->blocks.data(), norm.data(), n_blocks, block_size);

    double mse_std, mse_emp, pwr;
    leankv_codebook_mse(norm.data(), n_full_elems, gaussian,         n_levels, &mse_std, &pwr);
    leankv_codebook_mse(norm.data(), n_full_elems, out->global_levels, n_levels, &mse_emp, &pwr);

    double snr_std = 10.0 * std::log10(pwr / (mse_std + 1e-20));
    double snr_emp = 10.0 * std::log10(pwr / (mse_emp + 1e-20));
    double gain    = snr_emp - snr_std;

    std::fprintf(stderr,
        "leankv-calib: TQ%d fit on %zu samples (%d iters): "
        "SNR std=%.2f emp=%.2f gain=%+.2f dB\n",
        bits, n_full_elems, iters, snr_std, snr_emp, gain);

    // Tiny *global* gains are not worth the cache entry for non-outlier
    // layers — but we still want to run the per-layer detection below,
    // because models like Qwen3-8B have a bulk distribution that matches
    // Gaussian closely yet still have a heavy-center L0. We flag the
    // codebook as "use default for non-override layers" and use the real
    // fitted `global_levels` only as a reference for the ratio check.
    const bool global_useful = gain >= 0.3;
    if (!global_useful) {
        std::fprintf(stderr,
            "leankv-calib: global gain %.2f dB < 0.3 dB — "
            "marking global as default, still checking per-layer overrides\n", gain);
        out->flags |= leankv::CODEBOOK_FLAG_USE_DEFAULT;
    }

    // ── Per-layer outlier detection ──
    //
    // Pass 1: fit every layer and collect its inner level + the full level
    // vector. Pass 2: compute the median inner (robust to outliers — the
    // global fit itself is biased *toward* heavy-center layers, so it can't
    // serve as a reference). Pass 3: flag any layer whose inner deviates
    // more than 1.5x from the median. In practice this fires on L0 of
    // Qwen3-4B (2.35x) and Qwen3-8B (1.83x) — both genuinely heavy-center.
    out->n_overrides = 0;
    const float global_inner = out->global_levels[n_levels / 2];

    struct layer_fit {
        int il;
        float inner;
        float levels[leankv::CODEBOOK_MAX_LEVELS];
    };
    std::vector<layer_fit> fits;
    fits.reserve(s->max_layer_seen + 1);

    size_t elem_cursor = 0;
    for (int il = 0; il <= s->max_layer_seen && il < (int) s->layer_counts.size(); il++) {
        const size_t cnt = s->layer_counts[il];
        if (cnt < min_samples) { elem_cursor += cnt; continue; }
        const size_t il_blocks = (cnt / (size_t) block_size);
        if (il_blocks == 0) { elem_cursor += cnt; continue; }

        layer_fit lf{};
        lf.il = il;
        leankv_fit_empirical_codebook(
            s->blocks.data() + elem_cursor, il_blocks, block_size, n_levels, 40, lf.levels);

        if (leankv::codebook_levels_ok(lf.levels, n_levels)) {
            lf.inner = lf.levels[n_levels / 2];
            fits.push_back(lf);
        }
        elem_cursor += cnt;
    }

    if (fits.empty()) {
        std::fprintf(stderr, "leankv-calib: no per-layer fits succeeded\n");
    } else {
        // Median inner level across layers — robust baseline.
        std::vector<float> inners;
        inners.reserve(fits.size());
        for (const auto & f : fits) inners.push_back(f.inner);
        std::sort(inners.begin(), inners.end());
        const float median_inner = inners[inners.size() / 2];
        const float min_inner = inners.front();
        const float max_inner = inners.back();

        int min_il = -1, max_il = -1;
        for (const auto & f : fits) {
            if (f.inner == min_inner && min_il < 0) min_il = f.il;
            if (f.inner == max_inner && max_il < 0) max_il = f.il;
        }

        std::fprintf(stderr,
            "leankv-calib: per-layer inner: min=%.4f (L%d), median=%.4f, max=%.4f (L%d), "
            "global=%.4f, max/min=%.2fx\n",
            min_inner, min_il, median_inner, max_inner, max_il, global_inner,
            min_inner > 0 ? max_inner / min_inner : 0.0);

        // Flag outliers vs median. 1.5x is the empirical break — Qwen3 L0
        // sits at ~1.8-2.4x, normal layers cluster within ±10%.
        constexpr float kOutlierRatio = 1.5f;
        for (const auto & f : fits) {
            const float ratio = f.inner / median_inner;
            if (ratio > kOutlierRatio || ratio < 1.0f / kOutlierRatio) {
                if (out->n_overrides >= leankv::CODEBOOK_MAX_OVERRIDES) break;
                auto & ov = out->overrides[out->n_overrides++];
                ov.layer_idx = (uint32_t) f.il;
                std::memcpy(ov.levels, f.levels, sizeof(float) * n_levels);
                std::fprintf(stderr,
                    "leankv-calib:   layer %d outlier (inner %.4f vs median %.4f, ratio %.2f)\n",
                    f.il, f.inner, median_inner, ratio);
            }
        }
    }

    // If the global fit was flagged as default, zero the stored levels
    // now that we've finished using them as a reference. The loader
    // expects default-marker files to have zero global levels.
    if (!global_useful) {
        std::memset(out->global_levels, 0, sizeof(out->global_levels));
    }

    return true;
}

// ── Model fingerprint (Phase 7a) ─────────────────────────────────

uint64_t leankv_fingerprint_model(const struct llama_model * model) {
    if (!model) return 0;

    // Start with a stable seed: architecture string + corpus version.
    const char * arch_name = llama_model_arch_name(model->arch);
    if (!arch_name) arch_name = "unknown";
    uint64_t h = leankv::fnv1a_64(arch_name, std::strlen(arch_name));

    uint32_t cv = leankv::calib_corpus_version;
    h = leankv::hash_mix(h, leankv::fnv1a_64(&cv, sizeof(cv)));

    // hparams sample — cheap, no backend roundtrip.
    uint32_t hp[8] = {
        (uint32_t) model->hparams.n_embd,
        (uint32_t) model->hparams.n_layer,
        (uint32_t) model->hparams.n_head(0),
        (uint32_t) model->hparams.n_head_kv(0),
        (uint32_t) model->hparams.n_embd_head_k(0),   // layer-0 dims for model fingerprint
        (uint32_t) model->hparams.n_embd_head_v(0),   // (upstream made head dims per-layer)
        (uint32_t) model->hparams.n_vocab,
        (uint32_t) model->ftype,
    };
    h = leankv::hash_mix(h, leankv::fnv1a_64(hp, sizeof(hp)));

    // Mix in a handful of stable weight samples to distinguish finetunes
    // of the same shape. We pull 4 KB from tok_embd and 1 KB from a few
    // wk tensors via the backend API, which works on both CPU and GPU.
    auto mix_tensor = [&](const struct ggml_tensor * t, size_t n_bytes) {
        if (!t) return;
        uint32_t meta[6] = {
            (uint32_t) t->type,
            (uint32_t) t->ne[0], (uint32_t) t->ne[1],
            (uint32_t) t->ne[2], (uint32_t) t->ne[3],
            (uint32_t) std::strlen(t->name),
        };
        h = leankv::hash_mix(h, leankv::fnv1a_64(meta, sizeof(meta)));
        h = leankv::hash_mix(h, leankv::fnv1a_64(t->name, std::strlen(t->name)));

        const size_t n_total = ggml_nbytes(t);
        const size_t n = n_total < n_bytes ? n_total : n_bytes;
        if (n == 0) return;
        std::vector<uint8_t> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, n);
        h = leankv::hash_mix(h, leankv::fnv1a_64(buf.data(), n));
    };

    mix_tensor(model->tok_embd, 4096);
    const int n_layer = (int) model->layers.size();
    if (n_layer > 0) {
        mix_tensor(model->layers[0].wk, 1024);
        if (n_layer > 2) mix_tensor(model->layers[n_layer / 2].wk, 1024);
        mix_tensor(model->layers[n_layer - 1].wk, 1024);
    }
    return h;
}

// ── Auto-calibration (Phase 7a) ──────────────────────────────────

static int leankv_tq_bits(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_TQ2_0: return 2;
        case GGML_TYPE_TQ3_0: return 3;
        case GGML_TYPE_TQ4_0: return 4;
        default: return 0;
    }
}

static bool leankv_autocalib_env_enabled() {
    const char * env = std::getenv("LEANKV_CALIBRATION_AUTO");
    if (!env || !env[0]) return true; // default: on
    return !(env[0] == '0' && env[1] == 0);
}

// Run a single-sequence llama_decode on the embedded corpus, chunked by
// n_batch so we don't blow past the context's batch budget. Returns the
// number of tokens actually decoded (0 on failure).
static int leankv_autocalib_run_corpus(struct llama_context * lctx) {
    const struct llama_model * model = &lctx->model;
    const int n_ctx   = (int) lctx->cparams.n_ctx;
    const int n_batch = (int) lctx->cparams.n_batch;

    // Tokenize.
    const char * corpus = leankv::calib_corpus;
    const int corpus_len = (int) std::strlen(corpus);
    std::vector<llama_token> tokens(corpus_len + 16);
    int n_toks = llama_tokenize(model, corpus, corpus_len,
                                tokens.data(), (int32_t) tokens.size(),
                                /*add_special=*/true, /*parse_special=*/false);
    if (n_toks < 0) {
        tokens.resize(-n_toks);
        n_toks = llama_tokenize(model, corpus, corpus_len,
                                tokens.data(), (int32_t) tokens.size(),
                                true, false);
    }
    if (n_toks <= 0) {
        std::fprintf(stderr, "leankv-calib: tokenization failed (%d)\n", n_toks);
        return 0;
    }

    // Clamp to what the context can actually hold.
    if (n_toks > n_ctx) n_toks = n_ctx;
    tokens.resize(n_toks);

    // Feed in chunks of n_batch.
    llama_pos pos = 0;
    int decoded = 0;
    for (int off = 0; off < n_toks; off += n_batch) {
        const int n = std::min(n_batch, n_toks - off);
        llama_batch batch = llama_batch_get_one(tokens.data() + off, n, pos, 0);
        int rc = llama_decode(lctx, batch);
        if (rc != 0) {
            std::fprintf(stderr, "leankv-calib: llama_decode returned %d at off=%d n=%d\n", rc, off, n);
            break;
        }
        pos     += n;
        decoded += n;
    }
    return decoded;
}

// Scheduler eval callback used only during auto-calibration: routes
// captured k_cur tensors into the in-memory accumulator on `lctx`.
static int leankv_autocalib_sched_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    static const char * const prefix = "leankv_k_calib-";
    static const size_t prefix_len   = 15;
    if (ask) {
        return std::strncmp(t->name, prefix, prefix_len) == 0;
    }
    auto * lctx = static_cast<llama_context *>(user_data);
    if (!lctx || !lctx->leankv_calib) return true;
    const int il = std::atoi(t->name + prefix_len);
    leankv_calib_dump_tensor(lctx->leankv_calib, t, il);
    return true;
}

// Install a fitted codebook into the kernel-side runtime LUTs.
//
// - Global LUT is installed unless the codebook is the default marker (in
//   which case the runtime state is already the Gaussian default).
// - Per-layer overrides (Stage 4b) are installed regardless of the global
//   marker state — an outlier layer can still need its own LUT even when
//   the global fit was punted.
static void leankv_install_runtime_codebook(const leankv::codebook & cb) {
    if (cb.bits < 2 || cb.bits > 4) return;

    // ── Global install ────────────────────────────────────────────
    if (!cb.is_default_marker()) {
        int8_t lut_i8[leankv::CODEBOOK_MAX_LEVELS];
        float  lut_f [leankv::CODEBOOK_MAX_LEVELS];
        if (leankv::codebook_levels_to_i8(cb.global_levels, cb.n_levels, lut_i8) &&
            leankv::codebook_levels_to_f (cb.global_levels, cb.n_levels, lut_f))
        {
            ggml_tq_set_runtime_levels_i8(cb.bits, lut_i8);
            ggml_tq_set_runtime_levels_f (cb.bits, lut_f);
            std::fprintf(stderr,
                "leankv-calib: installed fitted global runtime LUT for bits=%d "
                "(levels[0..%d] = %+.4f, %+.4f, ..., %+.4f)\n",
                cb.bits, cb.n_levels - 1,
                cb.global_levels[0],
                cb.global_levels[1],
                cb.global_levels[cb.n_levels - 1]);
        } else {
            std::fprintf(stderr,
                "leankv-calib: WARNING: fitted global levels failed int8/float conversion "
                "for bits=%d — global runtime LUT unchanged\n", cb.bits);
        }
    } else {
        std::fprintf(stderr,
            "leankv-calib: codebook is default-marker for bits=%d — global runtime LUT unchanged\n",
            cb.bits);
    }

    // ── Per-layer overrides (Stage 4b) ────────────────────────────
    // These flow through the K-cache range registry populated by
    // llama_kv_cache_init. The FA helper's constructor looks up its
    // layer from its data pointer and picks up the override.
    for (int i = 0; i < cb.n_overrides; i++) {
        const leankv::codebook_layer_override & ov = cb.overrides[i];
        const int il = (int) ov.layer_idx;

        int8_t lut_i8[leankv::CODEBOOK_MAX_LEVELS];
        float  lut_f [leankv::CODEBOOK_MAX_LEVELS];
        if (!leankv::codebook_levels_to_i8(ov.levels, cb.n_levels, lut_i8) ||
            !leankv::codebook_levels_to_f (ov.levels, cb.n_levels, lut_f))
        {
            std::fprintf(stderr,
                "leankv-calib: WARNING: layer %d override failed int8/float conversion — skipping\n",
                il);
            continue;
        }
        ggml_tq_set_layer_override_i8(cb.bits, il, lut_i8);
        ggml_tq_set_layer_override_f (cb.bits, il, lut_f);
        std::fprintf(stderr,
            "leankv-calib: installed per-layer override L%d for bits=%d "
            "(levels[0..%d] = %+.4f, %+.4f, ..., %+.4f)\n",
            il, cb.bits, cb.n_levels - 1,
            ov.levels[0], ov.levels[1], ov.levels[cb.n_levels - 1]);
    }
}

void leankv_autocalibrate(struct llama_context * lctx) {
    if (!lctx) return;
    if (!leankv_autocalib_env_enabled()) return;

    const int bits = leankv_tq_bits(lctx->kv_self.type_k);
    if (bits == 0) return; // not a TQ K-cache

    const struct llama_model * model = &lctx->model;
    const char * arch_name = llama_model_arch_name(model->arch);
    if (!arch_name) arch_name = "unknown";
    const int n_layers = (int) model->hparams.n_layer;

    // Compute fingerprint and try the cache first.
    const uint64_t fp = leankv_fingerprint_model(model);
    if (fp == 0) {
        std::fprintf(stderr, "leankv-calib: fingerprint failed — skipping auto-calib\n");
        return;
    }
    const std::string path = leankv::codebook_cache_path(fp, bits);

    leankv::codebook cb{};
    if (leankv::codebook_load(path, fp, &cb)) {
        std::fprintf(stderr,
            "leankv-calib: using cached codebook '%s' (arch=%s, bits=%d, %d override%s%s)\n",
            path.c_str(), cb.arch, cb.bits, cb.n_overrides,
            cb.n_overrides == 1 ? "" : "s",
            cb.is_default_marker() ? ", default-marker" : "");
        leankv_install_runtime_codebook(cb);
        return;
    }

    // Cache miss — run a short warm-up inference with capture enabled.
    std::fprintf(stderr,
        "leankv-calib: no cached codebook for fp=%016llx arch=%s bits=%d — calibrating...\n",
        (unsigned long long) fp, arch_name, bits);

    // Stand up the in-memory accumulator.
    if (lctx->leankv_calib) {
        // Something else (e.g. LEANKV_CALIBRATION_DUMP=1) is already using
        // the hook — don't clobber it.
        std::fprintf(stderr,
            "leankv-calib: existing calib state present — skipping auto-calib\n");
        return;
    }
    lctx->leankv_calib = leankv_calib_init_in_memory();

    // Save any previously-installed eval callback so we can restore it.
    // We don't have a getter, so we assume none was set at this point
    // (auto-calib runs before the user ever calls llama_decode).
    ggml_backend_sched_set_eval_callback(lctx->sched, leankv_autocalib_sched_eval_cb, lctx);

    leankv_calib_set_runtime_capture(true);

    const int decoded = leankv_autocalib_run_corpus(lctx);

    leankv_calib_set_runtime_capture(false);
    ggml_backend_sched_set_eval_callback(lctx->sched, nullptr, nullptr);

    // Clear the KV cache state we just populated so the user's first real
    // decode starts from scratch.
    llama_kv_cache_clear(lctx);

    if (decoded == 0) {
        std::fprintf(stderr, "leankv-calib: warm-up decoded 0 tokens — aborting calib\n");
        leankv_calib_free(lctx->leankv_calib);
        lctx->leankv_calib = nullptr;
        return;
    }

    std::fprintf(stderr,
        "leankv-calib: warm-up decoded %d tokens, captured %d K-tensor record(s)\n",
        decoded, lctx->leankv_calib->n_records);

    // Fit.
    const int block_size = 32; // same QK as TQ2/TQ3/TQ4
    bool fit_ok = leankv_calib_fit_codebook(lctx->leankv_calib,
                                            bits, block_size,
                                            arch_name, n_layers, &cb);
    cb.fingerprint = fp;

    if (!fit_ok) {
        std::fprintf(stderr,
            "leankv-calib: fit declined — persisting default marker so we don't retry\n");
        leankv::codebook_make_default(&cb, fp, bits, arch_name, n_layers);
    }

    if (!leankv::codebook_save(path, cb)) {
        std::fprintf(stderr, "leankv-calib: WARNING: failed to save '%s'\n", path.c_str());
    }

    // Install into the kernel-side runtime LUT so the user's first real
    // decode sees the fitted values. Safe to call after save even if save
    // failed — the runtime install is a pure in-process op.
    leankv_install_runtime_codebook(cb);

    leankv_calib_free(lctx->leankv_calib);
    lctx->leankv_calib = nullptr;
}
