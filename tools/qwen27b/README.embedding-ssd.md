# Qwen embedding rows on SSD (development only)

Off by default. This changes the existing Qwen HIP backend only. It does not
change the output head, Flash-Next/DeepSeek backends, quantization arithmetic,
or NPU split path. Device identity/RAM/speed qualification is required before
promotion; CPU tests and host syntax checks are not that qualification.

## Export, then enable

With `GUFO_BUILD_TOOLS=ON`, build the CPU exporter explicitly:

```sh
cmake --build build/release --target export_embedding_rows --parallel 1
build/release/export_embedding_rows /path/to/model.gguf /path/to/model.embrows
```

- Export the actual target GGUF, not the drafter. Split GGUF input uses its
  normal first-shard path.
- Destination must not exist. Export publishes a completed temporary file
  atomically without replacing any existing file.
- Export and a cold canonical model-identity lookup read model data. Do not run
  them during a restricted/timed GPU window without the window owner's go.
- Current record format supports Q4_K token rows fitting one 4 KiB record.
  The output head must be separate; tied/overlapping embedding storage declines.

Launch the usual command with:

```sh
GUFO_DEV_EMBED_SSD=/path/to/model.embrows \
GUFO_DEV_EMBED_SSD_CACHE_MB=16 \
  build/release/gufo <usual arguments>
```

`GUFO_DEV_EMBED_SSD_CACHE_MB` is positive integer MiB; default 16. It bounds
encoded record bytes, excluding LRU metadata and executor staging. Cache
allocation grows on demand. Target, DFlash/DSpark and MTP share the row store;
each executor has private packed-row staging. Prefill deduplicates token IDs
and joins eight pread workers before the gather. Decode uses synchronous pread.
No asynchronous prefetch or hidden-I/O guarantee is claimed.

## Diagnostics and failure behavior

- `active=1` is logged only after target weight spans are loaded/remapped.
- Missing/ineligible/stale file, unsupported direct I/O, invalid cache setting,
  tied head or aliased storage: `active=0 resident_fallback=1 reason=...`, followed
  by the original resident load. Check this log before claiming SSD mode ran.
- After activation, incomplete reads and row ID/CRC failures throw; they do not
  fall back to stale/zero rows or change modes inside a graph.
- The file is immutable while open. Re-export to another path for another model.
- Allocation accounting is not a physical RAM measurement. Qualification must
  check system MemAvailable, repeated changing-token graph replay, greedy text
  identity with NPU off, and matched prose/code/json timing.

## File format v1 (Linux x86-64 little-endian)

- Header page: 4,096 bytes. Fixed-width fields: magic `GUFOEMB1`, version,
  GGML type, hidden width, vocabulary size, encoded row bytes, shard index,
  tensor file offset, canonical model SHA256 (64 hex bytes), aggregate original
  row SHA256 (64 hex bytes), header CRC32C. Remaining bytes are zero.
- Records: 4,096 bytes at `(token_id + 1) * 4096`. CRC32C at bytes 0–3; token ID
  at 4–7; original encoded GGUF bytes at 8 onward; zero padding. CRC covers bytes
  4–4095, including row ID and padding. These checks detect accidental corruption;
  they do not authenticate an untrusted row file.
- Opening validates header, extent, source identity and direct I/O without
  reading the entire row file. Each fetched record validates ID and CRC32C.
- Model identity reuses the existing cached GGUF SHA256 scheme; a cold lookup
  may stream the complete model. Per-row fetch does not compute SHA256.

For 5120 × 248320 Q4_K, the arithmetic is 2,880 bytes/row,
715,161,600 original tensor bytes and 1,017,122,816 row-file bytes including
header/padding. These are layout sizes, not measured RAM savings or speed.

## CPU checks

- CTest: `qwen_embedding_rows_test`, `qwen_embedding_rows_wiring_test`.
- The unit uses a small synthetic GGUF and actual O_DIRECT reads. Unsupported
  O_DIRECT is a failure, not a passing skip.
- Source wiring checks cover graph refresh, both loader modes and all gather
  consumers. They do not execute HIP kernels.
