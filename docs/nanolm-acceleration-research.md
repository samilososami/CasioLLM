# NanoLM on Casio fx-CG50: acceleration research

Date: 2026-08-16

## Baseline measured on the physical calculator

- Backend: NanoLM-25M-Instruct-v1.1, pruned 20K vocabulary, external Q4 weights.
- Model file: `NANOLM.Q4`, 11,537,200 bytes.
- Runtime geometry: 12 layers, hidden size 312, intermediate size 1092,
  12 attention heads, 4 KV heads, head dimension 26.
- A `hi` request currently expands to about 41-42 prompt tokens because of the
  fixed ChatML system prompt.
- The physical log after about two minutes stopped at prompt token 15/41,
  layer 6, with no generated token. This is approximately 8 seconds per input
  token and predicts several minutes to first output.

This is a performance failure, not a model-asset validation failure. TinyTalk
has already demonstrated that the chat UI, streaming, cancellation and external
model reads work end to end on the same calculator.

## What the present runtime is doing

For one transformer input token the current state machine performs roughly:

- 1 world switch to start the forward pass;
- 29 microsteps per layer x 12 layers = 348 microsteps;
- 1 final normalization microstep;
- later, about 19 more microsteps to scan the 20K output vocabulary.

That is about 350 gint/OS world switches per prompt token and about 369 steps
per generated token. Every `os_step()` opens and closes `NANOLM.Q4`, including
activation and residual stages that do not read the file.

The UI deliberately waits 9 ms after every inference microstep so that the
128 Hz keyboard scanner can observe normal key taps after returning from the
OS. At about 350 steps, this delay alone is approximately 3.15 seconds per
input token. The physical measurement is consistent with this: roughly 38% of
the observed time can be explained by the keyboard windows alone.

The Q4 file layout is also hostile to streaming:

- tensor order comes from lexicographic safetensors order (`0, 1, 10, 11, 2...`),
  not numeric layer or execution order;
- each tensor stores all fp16 scales separately from all packed nibbles;
- a matrix chunk therefore needs two positioned BFile reads and frequent seeks;
- the 12 KB buffer leads to roughly 120-125 BFile reads per layer, over 1,500
  reads for the transformer portion of one token.

The arithmetic is the other major bottleneck. The binary targets
`-m4-nofpu`; disassembly shows calls to software helpers such as `__mulsf3`,
`__addsf3` and `__floatsisf`. Although the Q4 x INT16 inner products are
integer, every 64-weight group converts and combines its subtotal with float
scales. The transformer has about 15.38 million matrix weights, which means
about 240,000 such scale groups per prompt token. The tied 20K output table adds
another 97,500 groups for each sampled token.

## Recommended runtime architecture

### 1. Precomputed fixed-prefix KV cache

The system prompt and the ChatML prefix before user text never change, and the
application is intentionally stateless. Generate their KV state on the PC with
the exact final quantized runtime and save it as a versioned external asset.

At startup of each request, load that state into the beginning of the KV cache
and tokenize only the user-dependent suffix. A Q8 prefix cache is expected to
be around 0.06-0.10 MB depending on the retained fixed prefix.

For a short request such as `hi`, this changes time-to-first-token work from
about 42 full forwards to roughly 7-9, while retaining the complete identity
prompt. The cache header must bind the prefix to the exact model SHA-256,
quantization version, token IDs and prefix length.

Prefix caching removes computation but does not remove the semantic effect of
the prompt. Local spot checks showed that NanoLM can respond less coherently
with the long branded identity prompt than with the short system wording it
was trained around. Before freezing `NANOLM.PFX`, compare the full identity
prefix against a canonical short assistant prefix over the complete prompt
battery. If the canonical prefix wins clearly, keep that prefix for inference
and answer only identity/author questions deterministically in the application;
this preserves the requested CasioLLM identity without degrading every reply.

### 2. Batched prompt prefill

Evaluate several prompt positions together per layer. The weights are read and
decoded once, then applied to several activation vectors. Causal attention is
still exact because each position only reads earlier KV entries.

A batch of 4 is conservative; 8 may fit after Q8 KV compression. Prefix caching
already removes most fixed tokens, so batch 3-4 is enough to cover many short
calculator prompts. This reduces repeated weight I/O during prompt evaluation;
it does not change the stateless behavior.

### 3. Q4-SH row-planar weight format

Create a new format in numeric layer and forward-execution order:

1. embedding table;
2. cached norms/metadata;
3. layer 0 Q/K/V/O, gate, up, down;
4. layer 1, and so on;
5. final norm.

Each matrix row should be contiguous. A proposed row contains:

- one fp16 base scale;
- one signed 7-bit/8-bit scale multiplier per 64-weight block;
- packed signed Q4 weights.

The per-block scales therefore become `base_scale * integer_subscale`. With a
global Q11 activation vector, every group and subscale operation can remain in
int32, followed by only one integer-to-float conversion and scale application
per output row. For the largest 1092-column row, the worst-case accumulator
still fits signed int32 when weights are restricted to -7..7.

This preserves almost all of the accuracy advantage of 64-weight scales while
removing most software-float operations. It is conceptually related to the
super-block scale design used by K-quants, but specialized for the SH4 and
NanoLM dimensions.

The row-planar layout also changes two separated scale/nibble reads into one
sequential read. With a 64 KB buffer, expected transformer BFile calls fall
from over 1,500 to roughly 150 per token before any larger batching.

### 4. Q8 KV cache and integer attention

The current fp16 KV cache consumes about 319,488 bytes:

`2 * 12 layers * 64 positions * 104 values * 2 bytes`.

Q8 K/V with one fp16 scale per KV head and position needs about 172 KB and
frees roughly 147 KB. This makes a larger I/O/projection buffer possible and
can extend context from 64 to 96 while still using less KV RAM than today.

Attention should then avoid float multiply-adds:

- quantize Q per head and compute Q x K as integer dot products;
- compute softmax in float or through a verified lookup table;
- combine `softmax_weight * value_scale` into a Q15 coefficient per position;
- accumulate Q15 coefficients x Q8 values in int32;
- convert once per output component.

This is more important as context grows because the current fp16 cache is
expanded to float and multiplied in software for every head, position and
dimension.

### 5. Coarser, cooperative scheduling

Never world-switch for activation-only or residual-only work. Two viable
schedulers should be benchmarked:

- responsive: load a complete projection, return to gint, and compute it in
  bounded row chunks while keyboard interrupts remain active;
- turbo: keep the Q4 file open and process a whole layer in one OS visit, then
  return to gint for display and F6 handling.

The first route needs a roughly 180 KB projection buffer, which becomes
plausible after Q8 KV. The largest gate/up/down matrices are about 177-179 KB in
the proposed layout. Q/K/V/O together are about 136 KB and can be grouped into
one attention block. The second route uses less RAM and fewer switches but has
longer F6 latency. Selection must be based on physical timings.

The selected output token should be displayed immediately after argmax. Its
forward pass is required to obtain the following token, not to display the
current one. The current runtime unnecessarily hides each chosen token until
that extra forward is complete.

### 6. SH4AL-DSP kernel

GCC accepts the DSP ISA but explicitly does not generate DSP instructions.
The hot dot-product loop therefore needs a small handwritten assembly kernel.

The calculator provides 4 KB ILRAM and 8 KB each of XRAM and YRAM with very
fast access. A practical kernel can:

- place its code in ILRAM;
- keep one Q11 activation vector in XRAM;
- unpack one Q4 row into signed 16-bit values in YRAM;
- use hardware-loop instructions plus `MAC.W` for each 64-value group;
- multiply group subtotals by integer subscales and finish with one float
  scaling operation per row.

Prompt batches should also have a C/assembly row-outer kernel that decodes one
weight row once and applies it to multiple activation vectors. Whether X/Y RAM
or the 32 KB operand cache wins for batched work must be measured on-device.

### 7. Secondary optimizations

- Replace 64-element SiLU microsteps with one stage and benchmark an fp16 LUT
  for SiLU/softmax before adopting approximate math.
- Compile and measure `-Os`, `-O2` and `-O3`; do not use `-Ofast` until outputs
  are checked because it relaxes floating-point semantics.
- Benchmark default F1, F4 (232 MHz CPU / 58 MHz bus) and F5 (189 MHz CPU /
  94 MHz bus). Overclocking must remain optional, restore the prior clock on
  exit, and only be considered after the non-overclocked runtime is stable.
- Keep the 20K vocabulary initially. Moving to 16K only shrinks the current
  model by about 0.66 MB and does not accelerate transformer prompt passes;
  prior local tests also showed a small relevance loss.
- Do not use raw flash addresses or private Fugue internals. Given the device's
  storage history, remain on supported BFile/filesystem operations.

## Verification gates

No optimization becomes the calculator default until it passes all gates:

1. Host-emulated logits and greedy responses against current Q4-20K.
2. The existing 75-prompt battery, including definitions such as `what is a
   dog?` and `what is water?`.
3. Exact tokenizer comparison for calculator-enterable text.
4. Physical profiler log with separate totals for world switching/open-close,
   BFile reads, each matrix family, attention, SiLU and vocabulary scan.
5. F6 cancellation and concurrent input tests during every scheduler mode.
6. Asset SHA/version validation and graceful rejection of mismatched prefix or
   model files.

## Implementation order

1. Build a NanoLM profiling add-in and capture the unmodified physical
   baseline with microsecond counters.
2. Coalesce activation stages, emit selected tokens immediately, and reduce
   world switches without changing model arithmetic.
3. Add fixed-prefix KV caching and batched prompt prefill.
4. Implement the row-planar Q4-SH converter plus a bit-exact host decoder.
5. Add Q11 matvec, Q8 KV and integer attention; run the 75-prompt comparison.
6. Add and benchmark the ILRAM/XRAM/YRAM assembly dot kernel.
7. Only then test LUT math, compiler levels and optional F4/F5 clock modes.

The realistic target for the first optimized build is not a promised token
rate. It is to turn the current multi-minute first-token latency into a
measurable tens-of-seconds or better path, then optimize from physical stage
timings. Prefix caching can provide the largest first-token gain; fixed-point
matvec/attention and the DSP kernel determine sustainable output throughput.

## Primary and implementation sources

- Renesas SH-4A Software Manual:
  https://www.renesas.com/en/document/mas/sh-4a-software-manual
- GCC SH target options:
  https://gcc.gnu.org/onlinedocs/gcc/SH-Options.html
- GCC optimization levels:
  https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
- llama.cpp quantized CPU dispatch and Q4 x Q8 kernels:
  https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ggml-cpu.c
- llama.cpp reference quantization code:
  https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
- llama2.c integer-activation inference reference:
  https://github.com/karpathy/llama2.c
- Hugging Face KV and prefix-cache documentation:
  https://huggingface.co/docs/transformers/kv_cache
- vLLM prefix-cache design:
  https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md
- Cardputer AI Q4 x Q8, row-planar storage and quantized KV reference:
  https://github.com/therezor/cardputer-ai
- gint/SH7305 memory, cache, profiler and DSP optimization reference:
  https://www.planet-casio.com/Fr/forums/topic16807-2-casm-optimiser-au-cycle-pres-la-reference.html
- gint high-resolution profiling introduction:
  https://www.planet-casio.com/Fr/forums/topic15796-last-libprof-mesure-de-temps-et-profilage-pour-gint.html
