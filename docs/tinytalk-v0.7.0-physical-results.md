# TinyTalk v0.7.0 physical results and lossless headroom

Date: 2026-08-16

## Calculator log

| Prompt | Prompt tokens | First token | Total | Switches | Internal units |
|---|---:|---:|---:|---:|---:|
| `hi` | 6 | 9.382 s | 34.296 s | 312 | 2447 |
| `can i tell you something` | 10 | 15.234 s | 24.117 s | 229 | 1815 |

The v0.7.0 eight-unit scheduler is active: `hi` performs 2447 legacy units in
312 bounded OS/storage visits. Its mandatory 9 ms keyboard windows account for
about 2.808 seconds, or 8.2% of the total. Compute and model reads now dominate.

## Lossless local experiments

A 49,152-byte I/O buffer instead of 24,000 bytes keeps all arithmetic and row
order unchanged. Five tested prompts remained byte-identical while cooperative
calls fell as follows:

| Prompt | 24 KB / batch 8 | 48 KB / batch 8 | Reduction |
|---|---:|---:|---:|
| `hi` | 312 | 242 | 22.4% |
| `i love you` | 221 | 177 | 19.9% |
| `what is a dog` | 756 | 580 | 23.3% |
| `how are you` | 361 | 282 | 21.9% |
| `i have something to tell you` | 244 | 199 | 18.4% |

Combining the 48 KB buffer with a 16-unit scheduler produced the same `hi`
reply in 127 cooperative calls and 1923 internal units. This should save at
least 1.665 seconds of explicit keyboard delay on that response, plus some
BFile/open-close overhead. It uses 25,152 additional heap bytes and increases
the worst-case F6 polling interval, so it requires a physical A/B test before
becoming the default.

## Further exact route

The fixed `User:` prefix can be represented by a version-bound KV cache. This
can remove roughly two forward passes from short prompts without changing
logits, vocabulary or context capacity. It needs a small exporter, exact-logit
host verification and a physical test.

The larger remaining speedup would require a bit-exact SH4AL DSP dot-product
kernel. This changes implementation rather than model capacity, but must be
checked against the C kernel before deployment.
