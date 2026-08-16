# CasioLLM NanoLM Q4-20K v0.7.0 hyper

Single-model Casio fx-CG50 release. Copy `CasioLLM-v0.7.0.g3a` to the
calculator as `CasioLLM.g3a`, then copy the six files under `assets/` to the
root of storage.

The runtime is stateless and streams generated tokens into the chat UI. F6
cancels generation. The 20-token fixed prefix is restored from `NANOLM.PFX`;
identity questions are handled deterministically.

See `docs/nanolm-v0.7.0-hyper.md` for the endian fix, exact optimizations and
verification evidence. TinyTalk v0.8.1 remains preserved as the stable
baseline in its own release directory.
