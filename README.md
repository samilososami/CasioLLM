# CasioLLM

Offline language-model runtime for the Casio fx-CG50, with a calculator-native
chat UI, token-by-token output, stateless prompts and F6 cancellation.

The project currently preserves three single-model milestones:

- **TinyTalk v1 Q4 v0.8.1**: stable physical baseline, approximately 2 MB of
  weights.
- **NanoLM CG4 Q4-20K v0.10.0 exact batched ultra**: current 25M-class
  candidate, 12.18 MB of externally streamed weights, four-position causal
  prefill and an exact SH4-specific runtime.
- **NanoLM Q4-20K v0.7.0 hyper**: preserved earlier milestone.

The application shows user messages in `#002A70`, model messages in `#468DF2`,
an animated thinking timer, streaming completion cursor, total response time
and a disabled/enabled paper-plane send button.

## TinyTalk physical results

The same `hi` reply was measured on the initial timestamped v0.7.0 runtime and
the final v0.8.1 runtime:

| Prompt | Build | First token | Total | Runtime switches | Internal units |
|---|---|---:|---:|---:|---:|
| `hi` | v0.7.0 | 9.382 s | 34.296 s | 312 | 2447 |
| `hi` | v0.8.1 | 5.695 s | 27.078 s | 57 | 1725 |

This is a 39.3% first-token improvement, a 21.0% total-time improvement and
81.7% fewer calculator OS/runtime transitions. The 50-prompt local regression
remained byte-identical.

Full methodology and the other physical prompts are in
[`docs/tinytalk-v0.8.1-final-results.md`](docs/tinytalk-v0.8.1-final-results.md).

## NanoLM exact batched ultra runtime

NanoLM v0.10.0 retains the accepted Q4-20K model behavior. Its exact runtime
changes include:

- Independent trimming of accidental trailing prompt spaces in both UI and
  model runtime.
- Four-position exact causal prefill: each matrix stream is reused for up to
  four new prompt positions.
- Native Q4-pair lookup, aligned unpacking, unrolled SH4 `MAC.W` and 32-byte
  `PREF` cache requests.
- A cached 20-token fixed prefix, reloaded for every stateless request.
- Coarser cooperative scheduling while retaining F6 cancellation.

All 50 varied reference prompts retain byte-identical text and token IDs.
Aggregate controlled host time falls from 46.095 s to 30.791 s, a 33.2%
reduction. Three complete reopen cycles and three cancel/reopen cycles pass
under ASan/UBSan. Host time is a regression signal; calculator timing is kept
in a separate physical-results table.

See
[`docs/nanolm-v0.10.0-exact-batched-ultra.md`](docs/nanolm-v0.10.0-exact-batched-ultra.md)
and
[`docs/performance-comparison-v0.9.0-v0.10.0.md`](docs/performance-comparison-v0.9.0-v0.10.0.md).

## Repository layout

- `runtime/`: current fxSDK/gint source and conversion/verification tools.
- `releases/`: self-contained calculator builds, external assets, checksums and
  preserved test evidence.
- `benchmarks/`: machine-readable comparison tables and complete prompt data.
- `docs/`: optimization research and physical measurements.
- `add-ins/`: recoverable Casio and game add-ins removed during the experiment.

Host-side Python dependencies used by the conversion and verification tools are
listed in `runtime/requirements-host.txt`.

## Installing a release

For TinyTalk, copy its `.g3a` as `CasioLLM.g3a` and the three `TINYTLK.*`
assets to the root of calculator storage.

For NanoLM v0.10.0, copy its `.g3a` as `CasioLLM.g3a` and all five
`NANOLM.*` assets from the release's `assets/` directory. Do not mix assets
from different releases. Verify with the included `SHA256SUMS` file.

## License and model attribution

This repository is distributed under GPL-3.0. NanoLM is based on
[`Mxode/NanoLM-25M-Instruct-v1.1`](https://huggingface.co/Mxode/NanoLM-25M-Instruct-v1.1),
whose model card declares GPL-3.0. See [`MODEL_ATTRIBUTION.md`](MODEL_ATTRIBUTION.md).
