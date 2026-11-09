---
name: transformers-upgrade
description: Update the pinned transformers-rwkv revision or its tokenizer and operator compatibility for irwkv.
---

# Update transformers-rwkv

1. Inspect the upstream revision and changes to RWKV configuration, tokenizer,
   generation, and the required FlashRWKV2 version. Keep the pinned dependencies
   in `requirements/rwkv.txt` mutually compatible.
2. Install the candidate in the project's `.venv`. Check model configuration,
   tokenizer loading, chat templates, and generation defaults using an existing
   RWKV checkpoint. Preserve the project's supported RWKV behavior.
3. Run the relevant `tests/transformers_utils`, `tests/tokenizers_`, and
   `tests/models/rwkv` tests. Compare real model outputs and log probabilities
   when the model path changes, and record evaluation results for changes that
   affect output, accuracy, or serving.

Use the repository's CUDA build and test instructions. Keep FlashRWKV2 changes
in its upstream project and report operator issues to the user.
