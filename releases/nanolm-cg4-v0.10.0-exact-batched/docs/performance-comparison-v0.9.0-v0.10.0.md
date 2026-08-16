# NanoLM performance comparison: v0.9.0 vs v0.10.0

Date: 2026-08-16

## Controlled exact host comparison

Both binaries use the same NanoLM CG4 Q4-20K assets and greedy decoding. Every
one of the 50 prompts generated identical text and identical token IDs. Times
below are milliseconds measured on the same host harness; they compare runtime
implementations and must not be read as calculator latency.

| Prompt | v0.9 first | v0.9 total | v0.10 first | v0.10 total | First reduction | Total reduction |
|---|---:|---:|---:|---:|---:|---:|
| `Hello! How are you today?` | 242 | 1007 | 85 | 664 | 64.9% | 34.1% |
| `I have something important to tell you.` | 304 | 960 | 117 | 640 | 61.5% | 33.3% |
| `I feel sad today. What could I do?` | 296 | 960 | 101 | 617 | 65.9% | 35.7% |
| `What is a dog?` | 187 | 960 | 54 | 664 | 71.1% | 30.8% |
| `What is a table?` | 195 | 976 | 62 | 664 | 68.2% | 32.0% |
| `What is the capital of France?` | 242 | 406 | 101 | 226 | 58.3% | 44.3% |
| `What is water?` | 171 | 960 | 54 | 687 | 68.4% | 28.4% |
| `What is 2 plus 2?` | 242 | 476 | 78 | 257 | 67.8% | 46.0% |
| `Why do people sleep?` | 210 | 992 | 62 | 718 | 70.5% | 27.6% |
| `Explain rain in simple words.` | 242 | 984 | 93 | 671 | 61.6% | 31.8% |
| `Tell me a joke.` | 187 | 968 | 62 | 671 | 66.8% | 30.7% |
| `Which planet do we live on?` | 257 | 1046 | 93 | 687 | 63.8% | 34.3% |

| Complete battery | v0.9.0 | v0.10.0 | Change |
|---|---:|---:|---:|
| Prompts | 50 | 50 | -- |
| Exact matching outputs | 50 | 50 | No quality change detected |
| Aggregate wall time | 46.095 s | 30.791 s | -33.2% |

## Physical calculator evidence

The only v0.9.0 physical request currently available contained an accidental
trailing space. It is retained as evidence but is not a clean A/B against
`hi`, because that space changes NanoLM tokenization and its reply.

| Prompt as tokenized | Build | First token | Observed total | Generated | Result |
|---|---|---:|---:|---:|---|
| `hi ` | v0.9.0 | 59.148 s | 255.437 s before F6 | 21 tokens | Incoherent deterministic branch |
| `hi` | v0.10.0 | Pending physical run | Pending physical run | Expected 26 tokens | `Hi! It's nice to meet you...` |

Based on the measured v0.9 transformer/vocabulary split and the exact
four-position prefill schedule, the pre-test engineering range for clean
v0.10.0 `hi` is roughly 22-30 seconds to first token and 210-240 seconds for
the full 26-token reply. This is explicitly an estimate. The table will be
updated with `NANOLM.LOG` after the installed build is run on the calculator.

## Historical TinyTalk comparison

This table is retained because it shows the progression from the initial
TinyTalk calculator runtime to the stable release. It is a different model and
must not be compared directly as a quality-equivalent NanoLM benchmark.

| Prompt | TinyTalk build | First token | Total | Reply |
|---|---|---:|---:|---|
| `hi` | v0.7.0 initial | 9.382 s | 34.296 s | `Hi! I'm Sarah. Nice to meet you!` |
| `hi` | v0.8.1 final | 5.695 s | 27.078 s | Same reply |

TinyTalk improved 39.3% to first token and 21.0% overall. Its v0.8.1 runtime
remains archived and unchanged after a proposed additional Q4 kernel failed
the speed-validation gate.

