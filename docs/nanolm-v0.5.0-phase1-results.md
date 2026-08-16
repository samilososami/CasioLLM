# NanoLM acceleration phase 1 - v0.5.0

Date: 2026-08-16

## Implemented

- Separate build backend; TinyTalk v0.4.3 and NanoLM v0.3.0 remain buildable.
- Fixed-prefix fp16 KV asset (`NANOLM.PFX`), 174,784 bytes.
- Prefix and v2 index both bound to Q4 SHA-256
  `0238295039aea89c677ff6608382ba5df3a9b4855eb026eb8cf4de381f6f5e03`.
- Stateless cache reload on every request.
- Eight old state-machine units per OS/gint switch and Q4 open.
- Token display immediately after argmax.
- Clean completion at the context boundary.
- Physical summary fields in `NANOLM.LOG`.
- Workspace-local libprof 2.4.0 CG library for later SH7305 kernels.

## Exact host-runtime regression for `hi`

Both tests execute `src/nanolm.c` itself against the calculator Q4 assets.
They are functional/call-count tests, not CG50 speed benchmarks.

| Metric | v0.3 legacy | v0.5 phase 1 |
|---|---:|---:|
| Prompt tokens processed | 42 | 7 plus 35 cached |
| Runtime calls until first visible token | 15,069 | 309 |
| Internal units until first token | 15,069 | 2,469 |
| Runtime calls for the context-limited response | 22,838 | 1,344 |
| Output | 22 tokens then context error | same 22 tokens, one final valid token, clean end |

The first 22 output tokens are byte-for-byte identical. The extra final token
in v0.5 was already selected from the last valid position; v0.3 hid it behind
an unnecessary next-token forward pass and then failed at the context limit.

The 9 ms keyboard window alone falls from about 135.6 seconds to 2.8 seconds
before the first token. On the real calculator, matrix compute and BFile I/O
still dominate and must be measured from the new log.

## Short-prompt candidate v0.5.1

A reproducible seven-way system-prompt comparison showed that the requested
long identity prompt strongly degrades this tiny model. The short prompt
`You are CasioLLM, a helpful assistant.` retained the useful greeting and was
materially better on several definitions. Identity and author questions are
therefore answered deterministically and streamed through the same UI.

The actual C runtime with Q4-I16 now returns for `hi`:

`Hi! It's nice to meet you. Is there something I can help you with, or would you like to chat?`

Its cache has 20 positions and occupies 99,904 bytes. The host runtime finishes
the 26-token reply cleanly in 1,531 coalesced calls. This remains a functional
test; CG50 time comes from the next physical run.

The proposed Q4-SH row-planar exporter was also implemented. Its 11,716,784-byte
model has slightly lower weight MSE than the current Q4, but the 10-prompt
Q11 smoke test produced zero byte-identical replies and mixed semantic changes.
It is preserved for further testing and is not yet the calculator default.

## Timing ranges before physical v0.5 measurement

These are engineering ranges, not promises. `Total` uses the observed greedy
`hi` reply: 11 output tokens for TinyTalk and a 23-token context-limited reply
for NanoLM with the current identity prompt.

| Runtime | First token | Total `hi` response | Evidence |
|---|---:|---:|---|
| TinyTalk v1 current | 10-20 s | 30-70 s | 731/2,447 scheduler calls; no timestamped physical run yet |
| NanoLM v0.3 current | 6-9 min | 10-15 min | physical run reached token 15/41 after about 2 min and another run exceeded 7.5 min without output |
| NanoLM v0.5 phase 1 | 25-60 s | 1.5-3.5 min | exact work reduction projected onto the physical v0.3 trace |
| NanoLM final Q4-SH/DSP target | 8-25 s | 35-100 s | target contingent on integer/DSP and row-planar physical benchmarks |
| TinyTalk final DSP target | 1-4 s | 4-12 s | target; TinyTalk already has row-planar Q4, Q8 activations and quantized KV |

Longer replies scale mainly with the generated-token rate and can exceed these
totals. The calculator log, not this table, becomes authoritative after the
first physical v0.5 run.

## Next implementation gates

1. Run v0.5 physically and collect `first_ms`, `total_ms`, `switches`, `units`.
2. Benchmark coalescing factors 4, 8 and 12 against F6 latency.
3. Export row-planar Q4-SH and run the 75-prompt semantic battery.
4. Add Q8 KV/integer attention and compare exact/semantic replies.
5. Implement and benchmark the SH4AL-DSP dot-product kernel.
6. Only then select compiler optimization and optional clock profiles.
