// LeanKV KV-importance calibration collector — see leankv-kvimp.h.

#include "leankv-kvimp.h"

#include "llama.h"
#include "llama-model.h"
#include "llama-arch.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <regex>
#include <fstream>
#include <algorithm>

namespace {

struct kvimp_layer_geom {
    int   n_head        = 0;
    int   n_head_kv     = 0;
    int   head_dim      = 0;
    bool  owns_kv       = true;
    int   owner         = -1;   // consumers: owning layer index
    int   reuse_count   = 1;    // owners: 1 + number of consumers
    bool  is_global     = false;
    float rope_fraction = 1.0f;
};

struct kvimp_accum {
    // indexed [kv_head] of the OWNING layer
    std::vector<double> v_importance, k_importance, v_var, k_var;
    // p-RoPE partitions (allocated only when rope_fraction < 1)
    std::vector<double> k_var_rope, k_var_pass, k_importance_rope, k_importance_pass;
    long long ntok = 0;         // advanced by the attn_k hook only
};

struct kvimp_ent {
    // indexed [kv_head] of the OWNING layer: running sum / count of per-row
    // attention entropies (row = one query token of one q head in the group)
    std::vector<double>    h_sum;
    std::vector<long long> h_cnt;
};

// "kq_soft_max_ext-<il>" / "kq_soft_max-<il>" -> il, else -1
static int kvimp_softmax_layer(const char * name) {
    const char * p = nullptr;
    if      (strncmp(name, "kq_soft_max_ext-", 16) == 0) p = name + 16;
    else if (strncmp(name, "kq_soft_max-",     12) == 0) p = name + 12;
    if (!p || !*p) return -1;
    char * end = nullptr;
    const long il = strtol(p, &end, 10);
    return (end != p && *end == '\0') ? (int) il : -1;
}

} // namespace

struct leankv_kvimp_state {
    std::map<int, kvimp_layer_geom> geom;
    std::map<int, kvimp_accum>      acc;
    std::map<int, kvimp_ent>        ent;
    std::regex  re{R"(blk\.(\d+)\.attn_(q|k|v|output)\.weight)"};
    std::string path     = "kv_stats.json";
    std::string ent_path = "kv_entropy.json";
    bool warned_geom_mismatch = false;
    bool warned_ent_shape     = false;
    bool checked_ent_axis     = false;

    kvimp_ent & ensure_ent(int layer) {
        auto it = ent.find(layer);
        if (it != ent.end()) return it->second;
        const kvimp_layer_geom & g = geom.at(layer);
        kvimp_ent e;
        e.h_sum.assign(g.n_head_kv, 0.0);
        e.h_cnt.assign(g.n_head_kv, 0);
        return ent.emplace(layer, std::move(e)).first->second;
    }

    kvimp_accum & ensure(int layer) {
        auto it = acc.find(layer);
        if (it != acc.end()) return it->second;
        const kvimp_layer_geom & g = geom.at(layer);
        kvimp_accum a;
        a.v_importance.assign(g.n_head_kv, 0.0);
        a.k_importance.assign(g.n_head_kv, 0.0);
        a.v_var.assign(g.n_head_kv, 0.0);
        a.k_var.assign(g.n_head_kv, 0.0);
        if (g.rope_fraction < 1.0f) {
            a.k_var_rope.assign(g.n_head_kv, 0.0);
            a.k_var_pass.assign(g.n_head_kv, 0.0);
            a.k_importance_rope.assign(g.n_head_kv, 0.0);
            a.k_importance_pass.assign(g.n_head_kv, 0.0);
        }
        return acc.emplace(layer, std::move(a)).first->second;
    }

    static int kv_of(int qh, const kvimp_layer_geom & g) {
        const int group = g.n_head / std::max(g.n_head_kv, 1);
        return group > 0 ? qh / group : 0;
    }
};

// ---------------------------------------------------------------------------
// geometry from hparams
// ---------------------------------------------------------------------------

static void kvimp_populate_geometry(leankv_kvimp_state & st, const llama_model & model) {
    const llama_hparams & hp = model.hparams;
    const int n_layer = (int) hp.n_layer;
    const int n_owned = hp.n_layer_kv_from_start > 0
        ? std::min<int>(n_layer, hp.n_layer_kv_from_start) : n_layer;

    for (int il = 0; il < n_layer; ++il) {
        const int nh  = (int) hp.n_head(il);
        const int nkv = (int) hp.n_head_kv(il);
        if (nh <= 0 || nkv <= 0) {
            continue;   // pure recurrent / DeltaNet layer: no attention KV
        }
        kvimp_layer_geom g;
        g.n_head    = nh;
        g.n_head_kv = nkv;
        g.head_dim  = (int) hp.n_embd_head_k(il);
        g.is_global = hp.swa_layers[il] == 0;
        g.owns_kv   = il < n_owned;
        const uint32_t n_rot_l = hp.rope_n_rot(il);
        g.rope_fraction = g.head_dim > 0
            ? std::min(1.0f, (float) n_rot_l / (float) g.head_dim) : 1.0f;

        if (!g.owns_kv) {
            // same rule as gemma4_mtp_target_kv_layer(): last owned layer of
            // the same type (sliding vs global).  [VERIFY vs the shared-layer
            // attention path on the first real E2B run]
            const bool sliding = hp.swa_layers[il] != 0;
            for (int t = n_owned - 1; t >= 0; --t) {
                if ((hp.swa_layers[t] != 0) == sliding) { g.owner = t; break; }
            }
            if (g.owner < 0) {
                fprintf(stderr, "leankv-kvimp: no same-type owner for shared layer %d — skipping\n", il);
                continue;
            }
        }
        st.geom.emplace(il, g);
    }

    // reuse_count on owners
    for (auto & kv : st.geom) {
        if (!kv.second.owns_kv && kv.second.owner >= 0) {
            auto it = st.geom.find(kv.second.owner);
            if (it != st.geom.end()) it->second.reuse_count += 1;
        }
    }

    int owned = 0, shared = 0;
    for (auto & kv : st.geom) (kv.second.owns_kv ? owned : shared) += 1;
    fprintf(stderr, "leankv-kvimp: geometry: %d attention layers (%d own KV, %d shared), arch=%s\n",
            owned + shared, owned, shared, llama_model_arch_name(model.arch));
}

// ---------------------------------------------------------------------------
// init / free
// ---------------------------------------------------------------------------

leankv_kvimp_state * leankv_kvimp_init(const llama_model & model) {
    const char * en = getenv("LEANKV_KVIMP");
    if (!en || en[0] == '0' || en[0] == '\0') return nullptr;

    auto * st = new leankv_kvimp_state();
    if (const char * p = getenv("LEANKV_KVIMP_PATH")) st->path = p;
    if (const char * p = getenv("LEANKV_KVIMP_ENTROPY_PATH")) {
        st->ent_path = p;
    } else if (st->path.size() > 5 && st->path.rfind(".json") == st->path.size() - 5) {
        st->ent_path = st->path.substr(0, st->path.size() - 5) + "_entropy.json";
    }
    kvimp_populate_geometry(*st, model);
    if (st->geom.empty()) {
        fprintf(stderr, "leankv-kvimp: no attention layers found — disabled\n");
        delete st;
        return nullptr;
    }
    fprintf(stderr, "leankv-kvimp: collecting KV-importance stats -> %s\n", st->path.c_str());
    return st;
}

// ---------------------------------------------------------------------------
// accumulation
// ---------------------------------------------------------------------------

static bool kvimp_read_f32(const ggml_tensor * t, std::vector<float> & out) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        out.resize(n);
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
        return true;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, ggml_nbytes(t));
        out.resize(n);
        ggml_fp16_to_fp32_row(tmp.data(), out.data(), n);
        return true;
    }
    return false;
}

// Attention softmax -> per-kv-head Shannon entropy over kv positions.
// Shape (from the non-FA build path, kq = mul_mat(k, q) then soft_max_ext):
//   ne[0] = n_kv (softmax axis), ne[1] = n query tokens, ne[2] = n q heads.
static void kvimp_accum_entropy(leankv_kvimp_state * st, const ggml_tensor * t, int layer) {
    auto git = st->geom.find(layer);
    if (git == st->geom.end()) return;
    const kvimp_layer_geom & g = git->second;

    // consumer layers (shared KV): fold into the owner, same rule as q/output
    const int target = g.owns_kv ? layer : g.owner;
    const kvimp_layer_geom & og = st->geom.at(target);
    if (!g.owns_kv && (g.n_head != og.n_head || g.n_head_kv != og.n_head_kv)) return;

    if (t->ne[2] != g.n_head || t->ne[3] != 1 || !ggml_is_contiguous(t)) {
        if (!st->warned_ent_shape) {
            fprintf(stderr, "leankv-kvimp: unexpected softmax shape [%lld,%lld,%lld,%lld] (n_head=%d) at layer %d — entropy skipped\n",
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], g.n_head, layer);
            st->warned_ent_shape = true;
        }
        return;
    }

    std::vector<float> buf;
    if (!kvimp_read_f32(t, buf)) return;

    const int64_t n_kv  = t->ne[0];
    const int64_t n_tok = t->ne[1];

    if (!st->checked_ent_axis) {
        // one-time empirical check that ne[0] really is the softmax axis
        double s = 0.0;
        for (int64_t j = 0; j < n_kv; ++j) s += buf[j];
        if (std::fabs(s - 1.0) > 0.05) {
            fprintf(stderr, "leankv-kvimp: softmax row sum %.4f != 1 at layer %d — check axis/sinks\n", s, layer);
        }
        st->checked_ent_axis = true;
    }

    kvimp_ent & e = st->ensure_ent(target);
    for (int h = 0; h < g.n_head; ++h) {
        const int kvh = leankv_kvimp_state::kv_of(h, g);
        if (kvh >= (int) e.h_sum.size()) continue;
        for (int64_t tk = 0; tk < n_tok; ++tk) {
            const float * p = buf.data() + (h * n_tok + tk) * n_kv;
            double H = 0.0;
            for (int64_t j = 0; j < n_kv; ++j) {
                if (p[j] > 0.0f) H -= (double) p[j] * std::log((double) p[j]);
            }
            e.h_sum[kvh] += H;
            e.h_cnt[kvh] += 1;
        }
    }
}

int leankv_kvimp_cb(leankv_kvimp_state * st, struct ggml_tensor * t, bool ask) {
    if (!st) return 0;

    const int sm_layer = kvimp_softmax_layer(t->name);
    if (sm_layer >= 0) {
        if (ask) return 1;
        kvimp_accum_entropy(st, t, sm_layer);
        return 0;
    }

    if (t->op != GGML_OP_MUL_MAT) return 0;

    const ggml_tensor * w = t->src[0];   // weight operand carries the clean name
    if (!w || !w->name[0]) return 0;

    std::cmatch m;
    if (!std::regex_search(w->name, m, st->re)) return 0;
    const int         layer = std::stoi(m[1].str());
    const std::string role  = m[2].str();   // q | k | v | output

    auto git = st->geom.find(layer);
    if (git == st->geom.end()) return 0;
    const kvimp_layer_geom & g = git->second;

    // consumer layers: only q/output, redirected into the owner's stats
    int target = layer;
    if (!g.owns_kv) {
        if (role != "q" && role != "output") return 0;
        target = g.owner;
    }
    if (ask) return 1;

    const kvimp_layer_geom & og = st->geom.at(target);
    if (!g.owns_kv &&
        (g.head_dim != og.head_dim || g.n_head != og.n_head || g.n_head_kv != og.n_head_kv)) {
        if (!st->warned_geom_mismatch) {
            fprintf(stderr, "leankv-kvimp: consumer layer %d geometry mismatch vs owner %d — skipping such layers\n",
                    layer, target);
            st->warned_geom_mismatch = true;
        }
        return 0;
    }
    kvimp_accum & a = st->ensure(target);

    // q:      OUTPUT (q activations)           -> t
    // output: INPUT  (attention out = V mix)   -> t->src[1]
    // k / v:  OUTPUT (the cached k/v)          -> t
    const ggml_tensor * src = (role == "output") ? t->src[1] : t;
    if (!src) return 0;

    std::vector<float> buf;
    if (!kvimp_read_f32(src, buf)) return 0;

    const int64_t nchan = src->ne[0];
    const int64_t ntok  = ggml_nelements(src) / std::max<int64_t>(nchan, 1);
    if (role == "k") a.ntok += ntok;

    std::vector<double> ss(nchan, 0.0);
    for (int64_t tk = 0; tk < ntok; ++tk) {
        const float * row = buf.data() + tk * nchan;
        for (int64_t c = 0; c < nchan; ++c) ss[c] += (double) row[c] * row[c];
    }

    const bool part      = og.rope_fraction < 1.0f;
    const int  rope_dims = part ? (int) lroundf(og.rope_fraction * og.head_dim) : og.head_dim;

    if (role == "q") {
        for (int64_t c = 0; c < nchan; ++c) {
            const int qh  = (int)(c / og.head_dim);
            const int cc  = (int)(c % og.head_dim);            // VERIFY-4
            const int kvh = leankv_kvimp_state::kv_of(qh, og);
            if (kvh >= (int) a.k_importance.size()) continue;
            a.k_importance[kvh] += ss[c];
            if (part) (cc < rope_dims ? a.k_importance_rope : a.k_importance_pass)[kvh] += ss[c];
        }
    } else if (role == "output") {
        for (int64_t c = 0; c < nchan; ++c) {
            const int qh  = (int)(c / og.head_dim);
            const int kvh = leankv_kvimp_state::kv_of(qh, og);
            if (kvh >= (int) a.v_importance.size()) continue;
            a.v_importance[kvh] += ss[c];
        }
    } else if (role == "v") {
        for (int64_t c = 0; c < nchan; ++c) {
            const int kvh = (int)(c / og.head_dim);
            if (kvh >= (int) a.v_var.size()) continue;
            a.v_var[kvh] += ss[c];
        }
    } else { // "k"
        for (int64_t c = 0; c < nchan; ++c) {
            const int kvh = (int)(c / og.head_dim);
            const int cc  = (int)(c % og.head_dim);            // VERIFY-4
            if (kvh >= (int) a.k_var.size()) continue;
            a.k_var[kvh] += ss[c];
            if (part) (cc < rope_dims ? a.k_var_rope : a.k_var_pass)[kvh] += ss[c];
        }
    }
    return 0;   // graph unmodified
}

// ---------------------------------------------------------------------------
// JSON out
// ---------------------------------------------------------------------------

void leankv_kvimp_free(leankv_kvimp_state * st) {
    if (!st) return;

    std::ofstream o(st->path);
    if (!o) {
        fprintf(stderr, "leankv-kvimp: cannot write %s\n", st->path.c_str());
        delete st;
        return;
    }
    o << "{\n  \"layers\": [\n";
    bool first = true;
    long long ntok_any = 0;
    for (const auto & kv : st->acc) {           // std::map -> sorted by layer
        const int layer       = kv.first;
        const kvimp_accum & a = kv.second;
        const kvimp_layer_geom & g = st->geom.at(layer);
        const double n = (double) std::max<long long>(a.ntok, 1);
        ntok_any = std::max(ntok_any, a.ntok);
        if (!first) o << ",\n";
        first = false;
        auto arr = [&](const char * key, const std::vector<double> & v) {
            o << "\"" << key << "\": [";
            for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << (v[i] / n);
            o << "]";
        };
        o << "    {\"layer\": " << layer
          << ", \"is_global\": "   << (g.is_global ? "true" : "false")
          << ", \"head_dim\": "    << g.head_dim
          << ", \"n_head\": "      << g.n_head
          << ", \"n_head_kv\": "   << g.n_head_kv
          << ", \"reuse_count\": " << g.reuse_count
          << ", \"rope_fraction\": " << g.rope_fraction
          << ", \"ntok\": "        << a.ntok
          << ", ";
        arr("v_importance", a.v_importance); o << ", ";
        arr("k_importance", a.k_importance); o << ", ";
        arr("v_var", a.v_var);               o << ", ";
        arr("k_var", a.k_var);
        if (!a.k_var_rope.empty()) {
            o << ", "; arr("k_var_rope",        a.k_var_rope);
            o << ", "; arr("k_var_pass",        a.k_var_pass);
            o << ", "; arr("k_importance_rope", a.k_importance_rope);
            o << ", "; arr("k_importance_pass", a.k_importance_pass);
        }
        o << "}";
    }
    o << "\n  ]\n}\n";
    fprintf(stderr, "leankv-kvimp: wrote %zu layers (ntok=%lld) -> %s\n",
            st->acc.size(), ntok_any, st->path.c_str());

    // attention-entropy JSON (kv_bit_allocator.py --entropy schema); k == v:
    // the attention distribution is one signal for both caches
    if (st->ent.empty()) {
        fprintf(stderr, "leankv-kvimp: no attention softmax seen — entropy file not written (flash attention never materializes kq_soft_max; calibrate with -fa off)\n");
    } else {
        std::ofstream eo(st->ent_path);
        if (!eo) {
            fprintf(stderr, "leankv-kvimp: cannot write %s\n", st->ent_path.c_str());
        } else {
            eo << "{\n";
            bool efirst = true;
            for (const auto & kv : st->ent) {   // std::map -> sorted by layer
                const kvimp_ent & e = kv.second;
                std::string lst;
                for (size_t i = 0; i < e.h_sum.size(); ++i) {
                    char b[32];
                    snprintf(b, sizeof(b), "%s%.6g", i ? "," : "",
                             e.h_sum[i] / (double) std::max<long long>(e.h_cnt[i], 1));
                    lst += b;
                }
                if (!efirst) eo << ",\n";
                efirst = false;
                eo << "  \"" << kv.first << "\": {\"k\": [" << lst << "], \"v\": [" << lst << "]}";
            }
            eo << "\n}\n";
            fprintf(stderr, "leankv-kvimp: wrote attention entropy for %zu layers -> %s\n",
                    st->ent.size(), st->ent_path.c_str());
        }
    }
    delete st;
}
