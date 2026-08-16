# TinyTalk v0.8.0 extreme

Date: 2026-08-16

## Lossless changes

- I/O buffer: 24,000 to 49,152 bytes.
- Cooperative batch: 8 to 128 legacy units.
- Exact cached prefix: token 7046 (`User`) and token 25 (`:`).
- Prefix asset: `TINYTLK.PFX`, 2,352 bytes, bound to the model geometry,
  model size, fixed model fingerprint and prefix-token hash.
- Missing or invalid prefix assets fall back to normal uncached inference.

No weight, quantizer, tokenizer, context, activation, attention, sampler or
generated-token rule was changed.

## Regression

Fifty varied prompts produced zero byte differences against v0.7.1. Total
cooperative calls fell from 20,549 to 1,005 (95.11%). For `hi`, runtime work
changes from 312 switches / 2447 internal units to 15 switches / 1725 units.

Three full shutdown/reopen cycles and three first-token cancel/reopen cycles
pass under AddressSanitizer with leak detection.

## F6 bound

The previous physical log implies roughly 12.9 ms average per old unit. The
larger buffer makes units somewhat coarser; a 128-unit block is projected near
2 seconds and within the user-approved 3-second cancellation budget. Physical
v0.8.0 logging is authoritative.

## Identity prompt decision

`You are CasioLLM.` was tested and rejected. It increased prompt work and
degraded answers (`who are you` invented a lawyer identity; `how are you`
regressed to `I am.`) without reliably adopting the requested name. Identity
should be handled deterministically outside the model if requested.
