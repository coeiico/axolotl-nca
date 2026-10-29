# axolotl-nca

**Growing neural cellular automata.** One small learned rule, applied at every pixel, that grows a shape from a single seed and regrows it after damage.

Part of the **Latent Space Project**: 18 small PyTorch repos that test one idea, that neural networks encode concepts and strategy in their latent space, and that we can read and edit those representations. Everything trains on a single consumer GPU or a laptop.

| Roadmap | Phase | Depends on | Status |
| --- | --- | --- | --- |
| 16 of 18 | Side quest, any time | Nothing | Not started (scaffolded) |

## Question

Can one small learned rule, applied identically at every pixel, grow a whole shape from a single seed and regrow it after damage?

## Plan

- **Architecture:** each cell holds 16 channels (RGBA plus 12 hidden). Fixed 3×3 Sobel and identity filters let it sense its neighbours. Two 1×1 convolutions (128 hidden, ReLU) compute an update. Cells update at random, and cells with low alpha count as dead.
- **Data:** one small RGBA target image (about 40×40) that the owner chooses or draws.

## Evaluation protocol

The target is one image shared by training and evaluation; this tests growth and recovery
for that target, not generalisation to unseen target images. Use separate rollout seeds
and damage conditions for development and final reporting, keep the final conditions out
of tuning, and report long-run stability as well as recovery. Record the target image's
hash with each run.

## Experiments

- Growth from a seed.
- Stability over long runs using a sample pool.
- Regeneration by damaging cells during training.
- An interactive demo (a Hugging Face Space) where the viewer erases part of the shape.

## Done when

- The shape grows reliably and regrows after damage, with a GIF in the README.

Plus the definition of done every Latent Space Project repo shares:

Apply this checklist with the project-specific requirements and exceptions in
[CLAUDE.md](CLAUDE.md#project-specific-scope).

- `just quick` runs end to end on CPU in under 5 minutes, as a smoke test.
- The full run fits on one consumer GPU (8–12 GB VRAM) and auto-detects cuda, mps or cpu.
- Each run saves weights, `metrics.json` and its plots to `runs/<run-name>/`.
- This README shows the headline result with plots, compares against baselines, and states negative results plainly.
- Tests cover data generators, tensor shapes and one tiny overfit check.

## Notes

- A side quest that can run at any point in the roadmap. Put the target image in `assets/`.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Everything else, Python 3.12 included, comes from uv.

```sh
just setup              # uv sync, then install the pre-commit hooks
just quick              # CPU smoke test, under 5 minutes
just train              # full run; flags pass through, e.g. --run-name wd1.0-seed0
just analyze <run>      # baselines, probes, interventions and plots
just test               # pytest
just check              # ruff and basedpyright, as in CI
```

Copy `.env.example` to `.env` (gitignored) for `HF_TOKEN` and `HF_USER`.

## Layout

```
axolotl-nca/
├── README.md                goal, plan, results with plots
├── CLAUDE.md                instructions for coding agents; AGENTS.md points here
├── .agents/                 instructions/ (Python formatting, commits) and plans/
├── pyproject.toml           metadata, dependencies, tool settings
├── uv.lock                  committed lockfile
├── .python-version          pinned by uv
├── justfile                 every routine command
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/axolotl_nca/
│   └── model.py, data.py, analysis.py, utils.py
├── train.py                 CLI; --quick runs the CPU smoke test
├── predict.py               load weights and run inference
├── analyze.py               baselines, probes, interventions, plots
├── notebooks/               marimo notebooks for exploration
├── tests/
└── runs/                    outputs, gitignored
```

## Results

_No results yet._ The headline result with plots, the comparison against baselines, and any negative results go here.

## Sources

- Mordvintsev et al. 2020, Growing neural cellular automata

## License

MIT
