# Slime runtime snapshot

The `slime` and `slime_plugins` packages come from the pinned runtime used by the paper (base commit in `../locks/stack.json`). The included files retain local model, agent, context-chain and checkpoint adaptations. This is a runtime source snapshot, not an unmodified upstream checkout. The release additionally removes unused GPU-lending/durable-queue RPCs and names the shared HF export helper `rsi_training.weight_export`.

Upstream documentation, examples, tests, datasets and repository metadata are excluded. The upstream Apache-2.0 license is retained in `slime/LICENSE`. Install dependencies in the matching GPU runtime and provide Megatron-LM separately.
