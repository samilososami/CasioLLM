# CasioLLM

Offline language-model runtime for the Casio fx-CG50, with a calculator-native
chat UI, token-by-token output, stateless prompts and F6 cancellation.

The project currently preserves two single-model releases:

- **TinyTalk v1 Q4 v0.8.1**: stable physical baseline, approximately 2 MB of
  weights.
- **NanoLM Q4-20K v0.7.0 hyper**: 25M-class experimental model, 11.54 MB of
  weights, with an SH4-specific exact runtime optimization pass.

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

## NanoLM hyper runtime

NanoLM v0.7.0 retains the previously selected Q4-20K weights. It does not use
the experimental Q4-SH re-quantization, because that changed model answers.
The exact runtime changes include:

- Correct little-endian decoding of the host-generated KV-prefix cache on the
  big-endian SH4.
- 49,152-byte storage buffer.
- SH4 `MAC.W` Q4 x INT16 dot products.
- Preloaded RMSNorm weights and RoPE denominators.
- Reuse of decoded GQA keys/values.
- Removal of repeated software-float softmax divisions.
- A cached 20-token fixed prefix, reloaded for every stateless request.

Ten representative old/new runtime replies are byte-identical. Three complete
reopen cycles and three cancel/reopen cycles pass under ASan/UBSan. Physical
SH7305 timing will be added after testing the installed v0.7.0 release.

See [`docs/nanolm-v0.7.0-hyper.md`](docs/nanolm-v0.7.0-hyper.md).

## Repository layout

- `runtime/`: current fxSDK/gint source and conversion/verification tools.
- `releases/`: self-contained calculator builds, external assets, checksums and
  preserved test evidence.
- `docs/`: optimization research and physical measurements.
- `add-ins/`: recoverable Casio and game add-ins removed during the experiment.

Host-side Python dependencies used by the conversion and verification tools are
listed in `runtime/requirements-host.txt`.

## Installing a release

For TinyTalk, copy its `.g3a` as `CasioLLM.g3a` and the three `TINYTLK.*`
assets to the root of calculator storage.

For NanoLM v0.7.0, copy its `.g3a` as `CasioLLM.g3a` and all five
`NANOLM.*` assets from the release's `assets/` directory. Do not mix assets
from different releases. Verify with the included `SHA256SUMS` file.

## License and model attribution

This repository is distributed under GPL-3.0. NanoLM is based on
[`Mxode/NanoLM-25M-Instruct-v1.1`](https://huggingface.co/Mxode/NanoLM-25M-Instruct-v1.1),
whose model card declares GPL-3.0. See [`MODEL_ATTRIBUTION.md`](MODEL_ATTRIBUTION.md).
