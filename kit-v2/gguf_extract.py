#!/usr/bin/env python3
"""Minimal GGUF reader: extract a named tensor's raw values as a numpy array.

Only what this study needs: parse the header + metadata KVs (to skip them) +
tensor infos, then read one small F32/F16 tensor by name. Not a general GGUF
library — just enough to pull `rope_freqs.weight` out of the E2B model so the
pre-RoPE study can reproduce gemma4 global-layer RoPE exactly.
"""
from __future__ import annotations

import struct
import sys

import numpy as np

# GGUF value type ids
GGUF_U8, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32, GGUF_F32, GGUF_BOOL, \
    GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64 = range(13)

_SCALAR_FMT = {
    GGUF_U8: ("<B", 1), GGUF_I8: ("<b", 1), GGUF_U16: ("<H", 2), GGUF_I16: ("<h", 2),
    GGUF_U32: ("<I", 4), GGUF_I32: ("<i", 4), GGUF_F32: ("<f", 4), GGUF_BOOL: ("<?", 1),
    GGUF_U64: ("<Q", 8), GGUF_I64: ("<q", 8), GGUF_F64: ("<d", 8),
}


class _R:
    def __init__(self, buf):
        self.b = buf
        self.o = 0

    def take(self, n):
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def gstr(self):
        n = self.u64()
        return self.take(n).decode("utf-8", "replace")

    def skip_value(self, vt):
        if vt in _SCALAR_FMT:
            _, sz = _SCALAR_FMT[vt]
            self.o += sz
        elif vt == GGUF_STRING:
            self.gstr()
        elif vt == GGUF_ARRAY:
            et = self.u32()
            cnt = self.u64()
            if et == GGUF_STRING:
                for _ in range(cnt):
                    self.gstr()
            else:
                _, sz = _SCALAR_FMT[et]
                self.o += sz * cnt
        else:
            raise ValueError(f"unknown gguf value type {vt}")


# ggml_type -> (numpy dtype, bytes/elem) for the plain types we might read
_GGML_TYPE = {0: (np.float32, 4), 1: (np.float16, 2)}


def read_tensor(path: str, want: str) -> np.ndarray:
    with open(path, "rb") as f:
        buf = f.read()
    r = _R(buf)
    magic = r.take(4)
    assert magic == b"GGUF", f"bad magic {magic!r}"
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    if version >= 3 or version == 2:
        pass  # v2/v3 both use u64 counts above
    # metadata
    alignment = 32
    for _ in range(n_kv):
        key = r.gstr()
        vt = r.u32()
        if key == "general.alignment" and vt == GGUF_U32:
            alignment = struct.unpack("<I", r.b[r.o:r.o + 4])[0]
            r.o += 4
        else:
            r.skip_value(vt)
    # tensor infos
    infos = {}
    for _ in range(n_tensors):
        name = r.gstr()
        ndim = r.u32()
        dims = [r.u64() for _ in range(ndim)]
        ttype = r.u32()
        offset = r.u64()
        infos[name] = (dims, ttype, offset)
    # data section starts at aligned offset
    data_start = r.o
    if data_start % alignment != 0:
        data_start += alignment - (data_start % alignment)
    if want not in infos:
        raise KeyError(f"{want} not in tensors; have e.g. "
                       f"{[n for n in infos if 'rope' in n.lower()]}")
    dims, ttype, offset = infos[want]
    npdt, esz = _GGML_TYPE[ttype]
    n = 1
    for d in dims:
        n *= d
    start = data_start + offset
    raw = buf[start:start + n * esz]
    arr = np.frombuffer(raw, dtype=npdt).astype(np.float32)
    return arr.reshape(dims[::-1])  # gguf dims are fastest-first


if __name__ == "__main__":
    path = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "rope_freqs.weight"
    a = read_tensor(path, name)
    print(f"{name}: shape={a.shape} dtype-as=f32")
    print("min/max/mean:", float(a.min()), float(a.max()), float(a.mean()))
    print("first 8:", a.reshape(-1)[:8])
    print("last 8:", a.reshape(-1)[-8:])
    print("all-ones?", bool(np.allclose(a, 1.0)))
