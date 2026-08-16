# TinyTalk v1 Q4: final extreme-optimization results

Date: 2026-08-16
Hardware: Casio fx-CG50, physical calculator
Stable release: `tinytalk-v1-q4-v0.8.1`

## Scope

The optimization keeps the same TinyTalk weights, tokenizer, Q4 arithmetic,
greedy sampler, context and response limits. It changes only how the runtime
loads data and schedules work:

- 49,152-byte storage buffer.
- Exact two-token `User:` KV-prefix cache (`TINYTLK.PFX`).
- 32 internal inference units per cooperative OS visit.
- Immediate prompt/thinking rendering before inference starts.
- Explicit little-endian decoding of the cached Q4 KV scales on the SH4.
- Clean model shutdown, tokenizer/buffer release and F6 cancellation.

The attempted 128-unit build was rejected after its host-generated prefix
scales were found to be interpreted in the wrong byte order on the big-endian
SH4. Version 0.8.1 fixes the cache format and uses 32-unit batches to keep F6
responsive.

## Physical comparison

`hi` is the prompt measured on both the initial timestamped v0.7.0 build and
the final v0.8.1 build.

| Prompt | Build | Reply | First token | Total | OS/runtime switches | Internal units |
|---|---|---|---:|---:|---:|---:|
| `hi` | v0.7.0 initial | `Hi! I'm Sarah. Nice to meet you!` | 9.382 s | 34.296 s | 312 | 2447 |
| `hi` | v0.8.1 final | `Hi! I'm Sarah. Nice to meet you!` | 5.695 s | 27.078 s | 57 | 1725 |

| Metric for `hi` | Improvement |
|---|---:|
| First-token latency | 39.3% lower |
| Total response time | 21.0% lower |
| OS/runtime switches | 81.7% fewer |
| Internal units | 29.5% fewer |

The second initial physical prompt was not repeated verbatim on v0.8.1, so it
is retained as baseline evidence rather than presented as a false A/B:

| Prompt | Build | First token | Total | Switches | Units |
|---|---|---:|---:|---:|---:|
| `can i tell you something` | v0.7.0 | 15.234 s | 24.117 s | 229 | 1815 |

## Final physical prompts

| Prompt | Reply | First token | Total | Rate after first token |
|---|---|---:|---:|---:|
| `hi` | `Hi! I'm Sarah. Nice to meet you!` | 5.695 s | 27.078 s | 2.138 s/token |
| `how are you` | `I'm doing well, thank you. How are you?` | 8.187 s | 32.382 s | 2.200 s/token |
| `im doing well too` | `That's good to be yourself.` | 9.492 s | 23.164 s | 2.279 s/token |

The 2.138-2.279 s/token range and 14.983-15.858 ms/internal-unit range are
stable. The longer first-token time follows the number of uncached input
tokens; there is no progressive runtime stall between prompts.

## Regression and integrity checks

- 50 varied prompts are byte-identical to v0.7.1.
- Cooperative calls fall from 20,549 to 3,939 (80.83%).
- Three complete shutdown/reopen cycles pass under AddressSanitizer.
- Three first-token cancel/reopen cycles pass under AddressSanitizer.
- The installed `.g3a`, Q4 weights, tokenizer and prefix cache match the
  release SHA-256 hashes byte for byte.
- The physical log contains no `ERROR`, `INVALID`, `NAN`, `FAIL` or unexpected
  cancellation entries.

## Conclusion

TinyTalk v0.8.1 is the stable calculator baseline. Remaining latency is the
expected cost of streaming model weights and executing matrix operations on
the SH7305; the logs do not show an avoidable scheduler or filesystem
bottleneck.
