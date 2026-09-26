# Packaging Notes

This package is intentionally narrower than the working experiment directory.

Included:

- Flat Python scripts needed to generate data, train RLOO + DReST LoRA adapters, parse answers, score option probabilities, and produce action-frequency plots.
- Exact paper-scale LLM datasets and the base pilot datasets needed by the deterministic regeneration scripts.
- Portable shell wrappers for the two main workflows: training and direct option-logprob evaluation.

Excluded:

- Cloud-provider launchers and recovery/watch scripts, because they contained machine-specific and account-specific assumptions.
- Private model/archive references, including old checkpoint storage locations.
- Raw run folders, logs, local mirrors, checkpoints, and final adapters, because they are large and not necessary for an anonymous code artifact.
- Paper source, bibliography, and figures, because this folder is meant to be merged into the separate anonymous code repository rather than used as the paper build folder.
- Notebook files, because the script versions are cleaner, easier to run non-interactively, and do not carry execution metadata.

