// LeanKV Phase 7: K-vector calibration dump implementation.
// See leankv-calib.h for format and env-var documentation.

#include "leankv-calib.h"

#include "ggml.h"
#include "ggml-backend.h"

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

bool leankv_calib_dump_tensor(leankv_calib_state * s, const struct ggml_tensor * t, int il) {
    if (!s || !s->file || !t) return false;
    if (s->max_records > 0 && s->n_records >= s->max_records) return false;

    const size_t n_bytes = ggml_nbytes(t);
    if (n_bytes == 0) return false;

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

    // Read tensor data via backend (handles GPU + CPU uniformly).
    std::vector<uint8_t> buf(n_bytes);
    ggml_backend_tensor_get(t, buf.data(), 0, n_bytes);
    std::fwrite(buf.data(), 1, n_bytes, s->file);
    std::fflush(s->file);

    s->n_records++;
    return true;
}
