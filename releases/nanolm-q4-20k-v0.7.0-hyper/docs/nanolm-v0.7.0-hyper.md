# NanoLM Q4-20K hyper runtime v0.7.0

Date: 2026-08-16
Target: Casio fx-CG50 / SH7305
Model: `Mxode/NanoLM-25M-Instruct-v1.1`, pruned 20K vocabulary, existing Q4

## Design decision

This build deliberately keeps the accepted 11,537,200-byte Q4-20K model.
The experimental row-planar Q4-SH re-quantization changed all ten initial
smoke-test replies and is therefore not installed. Version 0.7.0 optimizes the
runtime while preserving the existing quantized model and greedy outputs.

It is a NanoLM-only build. TinyTalk and every older NanoLM release remain in
the workspace/release archive but are not loaded into calculator RAM.

## Correctness fix

The previous `NANOLM.PFX` loader copied host-native `uint16` values directly
into the CG50's cache. The asset is little-endian; the SH4 is big-endian. This
made valid cached keys/values numerically corrupt on the calculator and could
produce nonsensical output.

The loader now reconstructs every fp16 cache value with explicit
little-endian decoding. The exporter likewise serializes every value
explicitly. Regenerating the 99,904-byte prefix produces a byte-identical file
with SHA-256 `3e4d4689e684aa305cc1dd3c5473337edd07aff34324542eddbaf5309d75ed50`.

## Exact runtime optimizations

- Storage buffer increased from 12,000 to 49,152 bytes.
- Q4 x INT16 dot products use the SH4 `MAC.W` instruction after unpacking each
  64-weight group. The final CG binary contains the instruction.
- All 25 RMSNorm tensors are read and converted once when the model opens,
  removing 25 reads and 7,800 fp16 conversions from every model token.
- The 13 RoPE denominators are calculated once instead of invoking `powf()`
  13 times for every processed token.
- Each GQA key/value head is converted from fp16 once and reused by its three
  query heads.
- Softmax probabilities are divided by their denominator once per position,
  rather than repeating the identical software-float division for all 26 head
  dimensions.
- The fixed 20-token system/user prefix is restored from `NANOLM.PFX` on every
  request, retaining stateless operation.
- The existing 8-unit cooperative batch is kept for F6 responsiveness.
- Timing starts at EXE/model submission, so physical first-token figures also
  include asset preparation, prefix loading and tokenization.
- Identity questions are answered deterministically through the same streamed
  UI, without asking the tiny model to infer its own name or author.

## Local verification

| Check | Result |
|---|---:|
| Representative old-runtime vs hyper-runtime replies | 10/10 byte-identical |
| `hi` output | `Hi! It's nice to meet you. Is there something I can help you with, or would you like to chat?` |
| Old host calls/internal units for `hi` | 1531 / 12063 |
| Hyper host calls/internal units for `hi` | 1477 / 11712 |
| Typical optimized host total for `hi` | about 1.1 s |
| Full reopen cycles under ASan/UBSan | 3/3 |
| First-token cancel/reopen cycles under ASan/UBSan | 3/3 |
| Prefix validator | valid; 20 positions, 12 layers, KV dimension 104 |

Host timing is only a regression signal and is not a calculator prediction.
The authoritative SH7305 figures must be collected from `NANOLM.LOG` after the
physical installation.

## Memory and files

| Component | Size |
|---|---:|
| Static BSS | 400,400 bytes |
| Dynamic storage buffer | 49,152 bytes |
| Add-in | 96,432 bytes |
| Q4 weights | 11,537,200 bytes |
| Prefix cache | 99,904 bytes |
| Tokenizer | 300,912 bytes |
| Tokenizer trie | 588,188 bytes |
| Tensor index | 1,816 bytes |
