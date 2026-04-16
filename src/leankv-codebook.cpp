// LeanKV Phase 7a: empirical codebook cache implementation.
// See leankv-codebook.h for the on-disk format and lifecycle.

#include "leankv-codebook.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace leankv {

// ── FNV-1a ────────────────────────────────────────────────────────

uint64_t fnv1a_64(const void * data, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    uint64_t h = 0xcbf29ce484222325ULL; // FNV offset basis
    const uint64_t prime = 0x100000001b3ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= prime;
    }
    return h;
}

uint64_t hash_mix(uint64_t a, uint64_t b) {
    // Boost-style mix.
    a ^= b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2);
    return a;
}

// ── Directory helpers ────────────────────────────────────────────

static bool path_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

static bool mkdir_p(const std::string & path) {
    if (path.empty()) return false;
    if (path_exists(path)) return true;

    // Walk the path and create each missing segment.
    std::string acc;
    for (size_t i = 0; i <= path.size(); i++) {
        if (i == path.size() || path[i] == '/') {
            if (i == 0) { acc = "/"; continue; }
            acc.assign(path, 0, i);
            if (!acc.empty() && acc != "/" && !path_exists(acc)) {
                if (::mkdir(acc.c_str(), 0755) != 0 && errno != EEXIST) {
                    std::fprintf(stderr,
                        "leankv-codebook: mkdir '%s' failed: %s\n",
                        acc.c_str(), std::strerror(errno));
                    return false;
                }
            }
        }
    }
    return true;
}

std::string codebook_cache_dir() {
    std::string base;
    if (const char * xdg = std::getenv("XDG_CACHE_HOME"); xdg && xdg[0]) {
        base = xdg;
    } else if (const char * home = std::getenv("HOME"); home && home[0]) {
        base = std::string(home) + "/.cache";
    } else {
        base = "/tmp";
    }
    std::string dir = base + "/leankv";
    mkdir_p(dir);
    return dir;
}

std::string codebook_cache_path(uint64_t fingerprint, int bits) {
    char name[64];
    std::snprintf(name, sizeof(name), "/%016llx_b%d.codebook",
                  (unsigned long long) fingerprint, bits);
    return codebook_cache_dir() + name;
}

// ── Validation ───────────────────────────────────────────────────

bool codebook_levels_ok(const float * levels, int n_levels) {
    if (!levels || n_levels < 2 || (n_levels & 1) != 0) return false;
    if (n_levels > CODEBOOK_MAX_LEVELS) return false;

    // No NaN / Inf.
    for (int i = 0; i < n_levels; i++) {
        if (!std::isfinite(levels[i])) return false;
    }
    // Monotonic increasing.
    for (int i = 1; i < n_levels; i++) {
        if (!(levels[i] > levels[i - 1])) return false;
    }
    // Outermost should be ±1 (within float roundoff of the fit).
    if (std::fabs(levels[0] + 1.0f) > 1e-5f) return false;
    if (std::fabs(levels[n_levels - 1] - 1.0f) > 1e-5f) return false;
    // Symmetric around 0.
    for (int i = 0; i < n_levels / 2; i++) {
        if (std::fabs(levels[i] + levels[n_levels - 1 - i]) > 1e-5f) return false;
    }
    // Minimum spread — reject degenerate "all levels clustered at ±1".
    float inner = levels[n_levels / 2];
    if (inner < 1e-4f || inner > 0.5f) return false;
    return true;
}

// ── Runtime-LUT conversion ──────────────────────────────────────

bool codebook_levels_to_i8(const float * in_f, int n_levels, int8_t * out_i8) {
    if (!in_f || !out_i8) return false;
    if (n_levels < 2 || n_levels > CODEBOOK_MAX_LEVELS) return false;

    // Canonical scale: levels ∈ [-1, 1] → int8 ∈ [-127, 127] via round(lvl * 127).
    // Matches the block-scale convention in the quantize / dequantize / vec_dot
    // paths (they divide d by 127.0f before reconstructing).
    std::memset(out_i8, 0, CODEBOOK_MAX_LEVELS);
    for (int i = 0; i < n_levels; i++) {
        const float v = in_f[i];
        if (!std::isfinite(v) || v < -1.0001f || v > 1.0001f) return false;
        float scaled = v * 127.0f;
        if (scaled > 127.0f)  scaled = 127.0f;
        if (scaled < -127.0f) scaled = -127.0f;
        // Round-half-away-from-zero, then clamp once more.
        const int32_t r = static_cast<int32_t>(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
        out_i8[i] = static_cast<int8_t>(r);
    }
    return true;
}

bool codebook_levels_to_f(const float * in_f, int n_levels, float * out_f) {
    if (!in_f || !out_f) return false;
    if (n_levels < 2 || n_levels > CODEBOOK_MAX_LEVELS) return false;

    std::memset(out_f, 0, sizeof(float) * CODEBOOK_MAX_LEVELS);
    for (int i = 0; i < n_levels; i++) {
        const float v = in_f[i];
        if (!std::isfinite(v) || v < -1.0001f || v > 1.0001f) return false;
        out_f[i] = v;
    }
    return true;
}

// ── Default marker ───────────────────────────────────────────────

void codebook_make_default(codebook * cb,
                           uint64_t   fingerprint,
                           int        bits,
                           const char *arch,
                           int        n_layers) {
    if (!cb) return;
    *cb = codebook{};
    cb->version     = CODEBOOK_FILE_VERSION;
    cb->fingerprint = fingerprint;
    cb->bits        = bits;
    cb->n_levels    = 1 << bits;
    cb->n_layers    = n_layers;
    cb->flags       = CODEBOOK_FLAG_USE_DEFAULT;
    if (arch) {
        std::snprintf(cb->arch, sizeof(cb->arch), "%s", arch);
    }
    // global_levels is left zero — callers must honor the flag and
    // pull Gaussian constants instead.
}

// ── Lookup ──────────────────────────────────────────────────────

const float * codebook::levels_for_layer(int il) const {
    for (int i = 0; i < n_overrides; i++) {
        if (overrides[i].layer_idx == static_cast<uint32_t>(il)) {
            return overrides[i].levels;
        }
    }
    return global_levels;
}

// ── On-disk header (exactly 96 bytes, must match the format in .h) ──

namespace {

struct file_header {
    uint32_t magic;
    uint32_t version;
    uint64_t fingerprint;
    uint32_t n_levels;
    uint32_t n_overrides;
    uint32_t bits;
    uint32_t flags;
    char     arch[32];
    uint32_t n_layers;
    uint32_t reserved[3];
};
// 4+4 + 8 + 4+4+4+4 + 32 + 4 + 12 = 80 bytes, natural alignment preserved.
static_assert(sizeof(file_header) == 80, "codebook file header layout changed");

} // namespace

// ── Read ────────────────────────────────────────────────────────

bool codebook_load(const std::string & path,
                   uint64_t             expected_fingerprint,
                   codebook *           out) {
    if (!out) return false;
    std::FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false; // missing cache is normal, not an error.

    file_header h{};
    if (std::fread(&h, sizeof(h), 1, f) != 1) {
        std::fclose(f);
        std::fprintf(stderr, "leankv-codebook: short header in '%s'\n", path.c_str());
        return false;
    }

    if (h.magic != CODEBOOK_FILE_MAGIC) {
        std::fclose(f);
        std::fprintf(stderr, "leankv-codebook: bad magic 0x%08x in '%s'\n",
                     h.magic, path.c_str());
        return false;
    }
    if (h.version != CODEBOOK_FILE_VERSION) {
        std::fclose(f);
        std::fprintf(stderr,
            "leankv-codebook: version mismatch (%u vs %u) in '%s' — will re-calibrate\n",
            h.version, CODEBOOK_FILE_VERSION, path.c_str());
        return false;
    }
    if (expected_fingerprint != 0 && h.fingerprint != expected_fingerprint) {
        std::fclose(f);
        std::fprintf(stderr,
            "leankv-codebook: fingerprint mismatch in '%s' — will re-calibrate\n",
            path.c_str());
        return false;
    }
    if (h.n_levels == 0 || h.n_levels > CODEBOOK_MAX_LEVELS) {
        std::fclose(f);
        std::fprintf(stderr, "leankv-codebook: bad n_levels=%u in '%s'\n",
                     h.n_levels, path.c_str());
        return false;
    }
    if (h.n_overrides > CODEBOOK_MAX_OVERRIDES) {
        std::fclose(f);
        std::fprintf(stderr, "leankv-codebook: too many overrides=%u in '%s'\n",
                     h.n_overrides, path.c_str());
        return false;
    }

    *out = codebook{};
    out->version     = h.version;
    out->fingerprint = h.fingerprint;
    out->bits        = static_cast<int>(h.bits);
    out->n_levels    = static_cast<int>(h.n_levels);
    out->n_layers    = static_cast<int>(h.n_layers);
    out->flags       = h.flags;
    std::memcpy(out->arch, h.arch, sizeof(out->arch));
    out->arch[sizeof(out->arch) - 1] = '\0';

    if (std::fread(out->global_levels, sizeof(float), h.n_levels, f) != h.n_levels) {
        std::fclose(f);
        std::fprintf(stderr, "leankv-codebook: truncated global levels in '%s'\n", path.c_str());
        return false;
    }

    // If the default-marker flag is set the levels are all zero, skip the shape check.
    if (!(h.flags & CODEBOOK_FLAG_USE_DEFAULT)) {
        if (!codebook_levels_ok(out->global_levels, out->n_levels)) {
            std::fclose(f);
            std::fprintf(stderr,
                "leankv-codebook: global levels failed sanity check in '%s' — will re-calibrate\n",
                path.c_str());
            return false;
        }
    }

    out->n_overrides = static_cast<int>(h.n_overrides);
    for (int i = 0; i < out->n_overrides; i++) {
        auto & o = out->overrides[i];
        uint32_t il = 0;
        if (std::fread(&il, sizeof(uint32_t), 1, f) != 1) {
            std::fclose(f);
            std::fprintf(stderr, "leankv-codebook: truncated override %d in '%s'\n",
                         i, path.c_str());
            return false;
        }
        o.layer_idx = il;
        if (std::fread(o.levels, sizeof(float), h.n_levels, f) != h.n_levels) {
            std::fclose(f);
            std::fprintf(stderr, "leankv-codebook: truncated override %d levels in '%s'\n",
                         i, path.c_str());
            return false;
        }
        if (!codebook_levels_ok(o.levels, out->n_levels)) {
            std::fclose(f);
            std::fprintf(stderr,
                "leankv-codebook: override %d (layer %u) failed sanity check in '%s'\n",
                i, o.layer_idx, path.c_str());
            return false;
        }
    }

    std::fclose(f);
    return true;
}

// ── Write (atomic: tmp + rename) ───────────────────────────────

bool codebook_save(const std::string & path,
                   const codebook &    cb) {
    if (cb.n_levels <= 0 || cb.n_levels > CODEBOOK_MAX_LEVELS) return false;
    if (cb.n_overrides < 0 || cb.n_overrides > CODEBOOK_MAX_OVERRIDES) return false;

    // Make sure the parent directory exists.
    std::string dir = path;
    size_t slash = dir.find_last_of('/');
    if (slash != std::string::npos) {
        dir.resize(slash);
        mkdir_p(dir);
    }

    std::string tmp = path + ".tmp";
    std::FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "leankv-codebook: cannot open '%s' for write: %s\n",
                     tmp.c_str(), std::strerror(errno));
        return false;
    }

    file_header h{};
    h.magic       = CODEBOOK_FILE_MAGIC;
    h.version     = cb.version ? cb.version : CODEBOOK_FILE_VERSION;
    h.fingerprint = cb.fingerprint;
    h.n_levels    = static_cast<uint32_t>(cb.n_levels);
    h.n_overrides = static_cast<uint32_t>(cb.n_overrides);
    h.bits        = static_cast<uint32_t>(cb.bits);
    h.flags       = cb.flags;
    std::memcpy(h.arch, cb.arch, sizeof(h.arch));
    h.arch[sizeof(h.arch) - 1] = '\0';
    h.n_layers    = static_cast<uint32_t>(cb.n_layers);

    auto fail = [&](const char * what) {
        std::fprintf(stderr, "leankv-codebook: write failed (%s): %s\n", what, std::strerror(errno));
        std::fclose(f);
        ::unlink(tmp.c_str());
        return false;
    };

    if (std::fwrite(&h, sizeof(h), 1, f) != 1)                                return fail("header");
    if (std::fwrite(cb.global_levels, sizeof(float), cb.n_levels, f) != (size_t) cb.n_levels)
        return fail("global levels");
    for (int i = 0; i < cb.n_overrides; i++) {
        const auto & o = cb.overrides[i];
        if (std::fwrite(&o.layer_idx, sizeof(uint32_t), 1, f) != 1)           return fail("override idx");
        if (std::fwrite(o.levels, sizeof(float), cb.n_levels, f) != (size_t) cb.n_levels)
            return fail("override levels");
    }

    std::fflush(f);
    std::fclose(f);

    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        std::fprintf(stderr, "leankv-codebook: rename '%s' → '%s' failed: %s\n",
                     tmp.c_str(), path.c_str(), std::strerror(errno));
        ::unlink(tmp.c_str());
        return false;
    }

    std::fprintf(stderr, "leankv-codebook: saved '%s' (%d levels, %d override%s)\n",
                 path.c_str(), cb.n_levels,
                 cb.n_overrides, cb.n_overrides == 1 ? "" : "s");
    return true;
}

} // namespace leankv
