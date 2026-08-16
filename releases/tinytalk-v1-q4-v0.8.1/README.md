# CasioLLM TinyTalk v1 Q4 v0.8.1

Hardware fix and responsiveness revision of v0.8.0:

- Prefix-cache scales are serialized and decoded explicitly as little-endian.
- Fixes the physical CG50 regression that emitted token 11 (comma) repeatedly.
- Prompt and `thinking...` render before the first inference block.
- Cooperative batch reduced from 128 to 32 for smoother animation and F6.
- 48 KB I/O buffer and exact two-token `User:` KV cache remain enabled.

Fifty prompts are byte-identical to v0.7.1. Cooperative calls remain 80.83%
below v0.7.1, and three shutdown/reopen plus cancel/reopen cycles pass ASan.
