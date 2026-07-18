#!/usr/bin/env python3
"""LeanKV Phase A: fit per-layer low-rank K projections from a calibration dump.

Reads a `.bin` file produced by Lean_llama.cpp with LEANKV_CALIBRATION_DUMP=1
(the leankv-calib format, same as scripts/analyze_k_calib.py in LeanKV), stacks
all post-RoPE K row-vectors per requested layer into [N, head_dim], runs SVD,
and keeps the top-r right singular vectors as an orthonormal projection basis
P [head_dim, rank]. The engine (src/leankv-lowrank.cpp) then replaces each K
vector k with its reconstruction P @ (P.T @ k) before the cache write.

Usage:
    python3 kit-v2/kv_lowrank_fit.py e2b_calib.bin --layers 4,9,14 --rank 224 -o kv_lowrank.bin
    # add --force to write the file even when a layer retains < 90% energy

Input format (little-endian; see src/leankv-calib.h):
    file header: u32 magic='KCAL' (0x4C41434B), u32 version=1
    each record: u32 rec_magic='LKCR' (0x52434B4C), u32 layer_idx, u32 dtype
                 (ggml_type: 0=f32, 1=f16), u32 n_dims, u32 ne[4], u32 nb[4],
                 u32 n_bytes, u8 data[n_bytes]
    tensor layout: ne[0]=head_dim, trailing dims flatten into the row count
                 (heads x tokens pack into dims 1+).

Output format (little-endian; MUST stay in sync with src/leankv-lowrank.h):
    file header: u32 magic='LKLR' (0x524C4B4C), u32 version=1, u32 n_entries
    each entry:  u32 layer_idx, u32 head_dim, u32 rank,
                 f32 P[head_dim * rank]  row-major [head_dim][rank]
                 (P[d*rank + j] = component d of the j-th top right singular
                  vector; columns of P are orthonormal)

Per-layer retained energy (sum of top-r squared singular values / total) is
printed; the tool exits 1 without writing when any requested layer retains
less than 90% unless --force is given.
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FILE_MAGIC = 0x4C41434B   # 'KCAL'
REC_MAGIC  = 0x52434B4C   # 'LKCR'

OUT_MAGIC   = 0x524C4B4C  # 'LKLR'
OUT_VERSION = 1

# ggml_type codes we might see for calibration tensors.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

MIN_ENERGY = 0.90


def read_records(path: Path):
    """Yield (il, np.ndarray shape [n_vecs, head_dim]) per record.

    Mirrors ~/LeanKV/scripts/analyze_k_calib.py:read_records exactly.
    """
    with path.open("rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        if magic != FILE_MAGIC:
            raise ValueError(f"bad file magic 0x{magic:08x} (expected 0x{FILE_MAGIC:08x})")
        if version != 1:
            raise ValueError(f"unsupported calib version {version}")

        while True:
            hdr = f.read(4 * 12)
            if not hdr:
                return
            if len(hdr) < 4 * 12:
                raise EOFError("truncated record header")
            rec_magic, il, dtype, ndims = struct.unpack("<IIII", hdr[:16])
            ne = struct.unpack("<IIII", hdr[16:32])
            nb = struct.unpack("<IIII", hdr[32:48])
            (n_bytes,) = struct.unpack("<I", f.read(4))
            data = f.read(n_bytes)
            if len(data) < n_bytes:
                raise EOFError("truncated record data")
            if rec_magic != REC_MAGIC:
                raise ValueError(f"bad record magic 0x{rec_magic:08x}")

            head_dim = ne[0]
            n_rows   = ne[1] if ndims >= 2 else 1
            # For ndims >= 3, flatten trailing dims into n_rows (heads x tokens)
            if ndims >= 3:
                n_rows *= ne[2]
            if ndims >= 4:
                n_rows *= ne[3]

            if dtype == GGML_TYPE_F32:
                arr = np.frombuffer(data, dtype=np.float32)
            elif dtype == GGML_TYPE_F16:
                arr = np.frombuffer(data, dtype=np.float16).astype(np.float32)
            else:
                raise ValueError(f"unsupported dtype {dtype}")

            expected = head_dim * n_rows
            if arr.size < expected:
                raise ValueError(f"data size {arr.size} < expected {expected}")
            arr = arr[:expected].reshape(n_rows, head_dim)
            yield il, arr


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fit per-layer low-rank K projections (LKLR file) from a KCAL dump")
    ap.add_argument("path", type=Path, help="calibration dump (KCAL format)")
    ap.add_argument("--layers", type=str, required=True,
                    help="comma-separated layer indices to fit, e.g. 4,9,14")
    ap.add_argument("--rank", type=int, required=True,
                    help="projection rank r (top-r right singular vectors)")
    ap.add_argument("-o", "--output", type=Path, required=True,
                    help="output LKLR file")
    ap.add_argument("--force", action="store_true",
                    help=f"write even if a layer retains < {MIN_ENERGY:.0%} energy")
    args = ap.parse_args()

    try:
        requested = [int(x) for x in args.layers.split(",") if x.strip()]
    except ValueError:
        print(f"error: bad --layers '{args.layers}'", file=sys.stderr)
        return 1
    if not requested:
        print("error: --layers is empty", file=sys.stderr)
        return 1
    if args.rank <= 0:
        print(f"error: --rank must be positive, got {args.rank}", file=sys.stderr)
        return 1

    per_layer: dict[int, list[np.ndarray]] = defaultdict(list)
    n_records = 0
    want = set(requested)
    for il, mat in read_records(args.path):
        if il in want:
            per_layer[il].append(mat)
        n_records += 1

    missing = [il for il in requested if il not in per_layer]
    if missing:
        print(f"error: layers {missing} not present in dump "
              f"({n_records} records read)", file=sys.stderr)
        return 1

    print(f"read {n_records} records from {args.path}")
    print(f"{'layer':>5} {'n_vecs':>8} {'dim':>5} {'rank':>5} {'energy':>8}")
    print("-" * 38)

    entries = []   # (il, head_dim, rank, P [head_dim, rank] f32 C-order)
    failed = []
    for il in requested:
        stacked = np.concatenate(per_layer[il], axis=0)  # [N, head_dim]
        n_vecs, dim = stacked.shape
        if args.rank > dim:
            print(f"error: layer {il}: rank {args.rank} > head_dim {dim}", file=sys.stderr)
            return 1
        if n_vecs < args.rank:
            print(f"error: layer {il}: only {n_vecs} vectors < rank {args.rank} "
                  f"(SVD cannot produce enough singular vectors)", file=sys.stderr)
            return 1

        # Thin SVD: stacked = U @ diag(sv) @ Vt, Vt rows are the right
        # singular vectors (orthonormal directions in K space).
        _, sv, vt = np.linalg.svd(stacked, full_matrices=False)

        energy_total = float(np.sum(sv ** 2))
        energy_kept  = float(np.sum(sv[: args.rank] ** 2))
        frac = energy_kept / energy_total if energy_total > 0 else 0.0

        # P [head_dim, rank]: column j = j-th top right singular vector.
        p = np.ascontiguousarray(vt[: args.rank].T, dtype=np.float32)
        assert p.shape == (dim, args.rank)

        print(f"{il:>5} {n_vecs:>8} {dim:>5} {args.rank:>5} {frac:>7.2%}")
        entries.append((il, dim, args.rank, p))
        if frac < MIN_ENERGY:
            failed.append((il, frac))

    if failed and not args.force:
        for il, frac in failed:
            print(f"error: layer {il} retains only {frac:.2%} < {MIN_ENERGY:.0%} "
                  f"energy at rank {args.rank}", file=sys.stderr)
        print("refusing to write projection file (use --force to override)", file=sys.stderr)
        return 1

    with args.output.open("wb") as f:
        f.write(struct.pack("<III", OUT_MAGIC, OUT_VERSION, len(entries)))
        for il, dim, rank, p in entries:
            f.write(struct.pack("<III", il, dim, rank))
            f.write(p.tobytes(order="C"))

    total_bytes = args.output.stat().st_size
    print(f"wrote {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} "
          f"to {args.output} ({total_bytes} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
