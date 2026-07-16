# Upstream ik_llama.cpp PR #2137 test — fullfp16 fix verification

**Date:** 2026-07-15
**Result: PASS**

Context: our upstream report ([ik_llama.cpp PR #2133](https://github.com/ikawrakow/ik_llama.cpp/pull/2133))
described Apple clang `-march=native` silently downgrading the target so the
`iqk` sources fail at compile time with
`always_inline 'vfmaq_f16' requires target feature 'fullfp16'`.
Upstream fix: [PR #2137](https://github.com/ikawrakow/ik_llama.cpp/pull/2137).

## Test setup

- Machine: Apple M2, macOS (Darwin 25.5.0)
- Toolchain: `Apple clang version 21.0.0 (clang-2100.1.1.101)`, target `arm64-apple-darwin25.5.0`
- Branch: `test-pr-2137` fetched directly via
  `git fetch https://github.com/ikawrakow/ik_llama.cpp.git pull/2137/head:test-pr-2137`
- Fresh build dir, **default flags only** — deliberately no `GGML_NATIVE=OFF`,
  no `GGML_ARCH_FLAGS`, no `-mcpu=` override:

```
cmake -B build-2137 -DGGML_METAL=ON
cmake --build build-2137 -j
```

## Results

- Configure: OK. Note it still selects `ARCH_FLAGS = -march=native`
  (`COMPILER_SUPPORTS_FP16_FORMAT_I3E` test fails, as before).
- Build: completed to 100% with **no** `fullfp16` errors. The fix therefore
  works at the source level in the `iqk` files, not by changing the arch flag.
  Only warnings: 3 unrelated `-Wtautological-constant-out-of-range-compare`
  (`STOP_TYPE_WORD` vs bool) in `examples/server/server-task.cpp`.
- Runtime sanity check:
  `./build-2137/bin/llama-cli -m qwen2.5-0.5b-instruct-q4_k_m.gguf -ngl 99 -p "hello" -n 16`
  generated cleanly on Metal — `FP16_VA = 1`, ~50 tok/s eval, ~63 tok/s prompt.

## Implication for Lean_llama.cpp

Our own tree fixed the same issue by switching to `-mcpu=native`
(commit 95c8d63c), which remains the semantically correct flag on Apple clang
(fullfp16 preserved via the CPU feature set). With #2137 merged upstream, the
`-mcpu=native` change becomes a correctness/consistency improvement rather than
a required workaround — worth mentioning in the #2137 comment but no longer a
blocker for default builds.
