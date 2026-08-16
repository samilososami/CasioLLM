# NanoLM CG4 v0.10.0: exact batched ultra runtime

Date: 2026-08-16  
Target: Casio fx-CG50 / SH7305  
Model: `Mxode/NanoLM-25M-Instruct-v1.1`, pruned 20K vocabulary, Q4 CG4

## Result

Version 0.10.0 preserves all 50 reference responses and token IDs while
reducing the aggregate host runtime from 46.095 s to 30.791 s, a 33.2%
reduction. The host timing is a controlled regression measurement, not a
prediction of SH7305 wall time. Physical figures are recorded separately.

The weights, tokenizer, greedy sampler, context, system-prefix KV cache and
response limit are unchanged. No approximate exponential, altered quantizer,
vocabulary pruning, output shortcut or `-ffast-math` option is used.

## User-input correctness fix

The UI and runtime now independently remove trailing ASCII spaces before
display, logging and tokenization. An all-space prompt is not submitted.

This fixes a real failure found in the v0.9.0 calculator log: the physical
`hi` request was actually logged and tokenized as `hi `, producing two user
tokens and a different, incoherent deterministic reply. In v0.10.0 both `hi`
and `hi ` produce the same one-token user suffix, hash `888cd50b`, output and
26 generated token IDs.

## Exact optimizations

### Four-position causal prompt prefill

Up to four uncached prompt positions are evaluated together. Each matrix row
is streamed and unpacked once, then applied independently to the four
activation vectors. Positions commit K/V entries in causal order; attention
for position N can only see positions at or before N. Decode remains scalar.

This is the main time-to-first-token improvement. For a clean `hi`, seven
uncached suffix positions are processed as two streamed batches instead of
seven separate full-model streams. All arithmetic for each position retains
the scalar runtime's operation order.

The implementation follows the same prefill/decode distinction exposed by
llama.cpp's batch API and benchmark. KV caching prevents recomputing the fixed
prefix, while batching reuses a weight read across several new prompt
positions.

### Exact SH4 Q4 inner loop

- A 256-entry native-endian lookup table expands both signed Q4 nibbles.
- An aligned 32-bit alias-safe store writes two unpacked `int16_t` weights.
- The SH4 `MAC.W` loop is unrolled eight times and returns the same exact
  signed `int32_t` subtotal as the scalar implementation.
- `PREF` requests the next 32-byte CG4 cache block.
- Aligned big-endian CG4 scale words are read directly as the same native
  float bit pattern, avoiding byte reconstruction and `memcpy` in the hot
  path.

Renesas documents `MAC.W` as a signed 16-by-16 multiply-accumulate and `PREF`
as a 32-byte cache-block prefetch. GCC accepts the SH4AL-DSP target but states
that it does not generate DSP instructions itself, which is why the hot
multiply-accumulate uses a small explicit assembly loop.

### Scheduling and memory

- Prompt-prefill work is coalesced at 11 stages per OS-world visit, roughly
  one transformer layer.
- Decode uses 64 internal units per cooperative visit.
- The storage buffer is 32,768 bytes.
- Static BSS is 446,704 bytes; BSS plus the storage buffer is 479,472 bytes,
  leaving headroom for stack and runtime allocations below the working
  512-KB envelope.
- F6 remains cooperative: it is observed whenever control returns to the UI.

## Rejected paths

- SH4AL-DSP X/Y-memory kernels were not enabled. They require `SR.DSP` and
  careful ownership of on-chip X/Y RAM; Casio OS calls also run inside the OS
  world, so adopting them without a dedicated save/restore protocol risks
  corrupting calculator state.
- Clock changes were not enabled. gint restores clock state across world
  switches, while model reads occur inside OS-world calls. Per-call clock
  mutation would be unsafe and could cost more than it saves.
- Approximate SiLU/softmax tables, a smaller vocabulary and re-quantization
  were rejected because the requirement is to preserve the accepted replies.
- A TinyTalk Q4 lookup/`MAC.W` prototype passed 50/50 equivalence but was
  almost twice as slow in the controlled host test because its Q8 layout
  required a separate unpack/widen path. It was reverted; TinyTalk v0.8.1
  remains the validated calculator build.

## Verification gates passed

| Gate | Result |
|---|---:|
| Varied prompt/token comparison with v0.9.0 | 50/50 exact |
| Aggregate host runtime | 46.095 s -> 30.791 s |
| Reduction | 33.2% |
| Clean `hi` output | 26/26 expected token IDs |
| `hi` vs `hi ` after normalization | Identical |
| ASan/UBSan shutdown and reopen | 3/3 |
| ASan/UBSan first-token cancel and reopen | 3/3 |
| CG binary contains `MAC.W` | Yes |
| Add-in name/version | `CasioLLM`, v0.10.0 |

The complete 50-prompt evidence is stored as
`benchmarks/nanolm-v0.9.0-v0.10.0-exact-50.json` in the repository and
release archive.

## Primary sources

- GCC SH options: https://gcc.gnu.org/onlinedocs/gcc/SH-Options.html
- GCC optimization options: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
- Renesas SH-4A Software Manual: https://www.renesas.com/en/document/mas/sh-4a-software-manual
- Renesas SH4AL-DSP Software Manual: https://www.renesas.com/en/document/mas/sh4al-dsp-software-manual
- llama.cpp batch API: https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h
- llama.cpp batched benchmark: https://github.com/ggml-org/llama.cpp/blob/master/examples/batched-bench/batched-bench.cpp
- llama.cpp quantized CPU implementation: https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cpu
- llama2.c quantized inference reference: https://github.com/karpathy/llama2.c
- Hugging Face cache explanation: https://huggingface.co/docs/transformers/cache_explanation

