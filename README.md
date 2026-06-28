# cs336 — Language Modeling from Scratch

Working through Stanford [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/),
building the full LLM stack from first principles — no high-level framework shortcuts.

## Scope

CS336 builds a language model end to end. Planned tracks (mapped to the course assignments):

| Track | Topic | Status |
|-------|-------|--------|
| 1 | Basics — BPE tokenizer, Transformer LM, training loop, optimizer | not started |
| 2 | Systems — kernels (Triton), FlashAttention, distributed data parallel | not started |
| 3 | Scaling laws — fit and extrapolate compute-optimal scaling | not started |
| 4 | Data — filtering, dedup, curation pipelines | not started |
| 5 | Alignment — SFT, DPO/RLHF | not started |

## Layout

```
cs336/
  tokenizer/   BPE training + encode/decode
  models/      Transformer LM components
  training/    optimizer, lr schedule, train loop, checkpointing
  data/        dataset loading + batching
  utils/       shared helpers
tests/         unit tests (run with pytest)
scripts/       CLI entry points (train, eval, tokenize)
notes/         derivations, experiment logs
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Constraints

The point of the course is to implement the primitives by hand. Use PyTorch tensors/autograd
and the standard library; avoid `transformers`, `nn.Transformer`, and other turnkey LM modules
for the core components being taught.
