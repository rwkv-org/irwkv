# Testing RWKV model changes

Run the relevant tests in `tests/models/rwkv/` and compare model outputs and
log probabilities with an authoritative RWKV implementation. Include evaluation
results when a change affects output, accuracy, or serving.

Measure performance with `irwkv bench`. Follow the repository agent
instructions for the model size, batch sizes, and prefill and decode metrics
when a custom benchmark is necessary. Record the hardware, model, configuration,
commands, and results with the change.
