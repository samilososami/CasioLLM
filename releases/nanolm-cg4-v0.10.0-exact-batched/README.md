# CasioLLM NanoLM CG4 v0.10.0

This is the exact batched ultra release for the Casio fx-CG50.

Copy `CasioLLM-v0.10.0.g3a` to the calculator storage root as
`CasioLLM.g3a`. Copy all five files from `assets/` to that same root. The
assets are byte-identical to v0.9.0; only the add-in runtime changed.

The application is stateless. It removes trailing spaces before tokenization,
streams generated text, accepts prompt typing while the model runs, and uses
F6 for cooperative cancellation.

Verification:

```sh
sha256sum -c SHA256SUMS
```

Technical details and the complete benchmark methodology are in `docs/`.
Machine-readable results are in `benchmarks/`.

