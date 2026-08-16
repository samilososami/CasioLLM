# Model attribution

## NanoLM

- Upstream model: `Mxode/NanoLM-25M-Instruct-v1.1`
- Upstream URL: <https://huggingface.co/Mxode/NanoLM-25M-Instruct-v1.1>
- Upstream author/account: Mxode
- Declared license: GPL-3.0
- Language: English
- Architecture: Mistral-compatible, 12 layers, hidden size 312

The calculator asset in this repository is a modified derivative:

- vocabulary pruned to 20,000 tokens;
- embeddings/output rows adjusted to that vocabulary;
- weights quantized to a custom symmetric Q4 format;
- tokenizer exported to calculator-specific table/trie files;
- a fixed-prefix KV cache generated for stateless inference.

The derivative is intended for research and experimentation on the Casio
fx-CG50. Model outputs can be inaccurate or nonsensical and should not be
treated as authoritative information.
