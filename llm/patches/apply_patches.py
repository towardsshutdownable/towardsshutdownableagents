#!/usr/bin/env python3
"""Apply the small edits to TRL 1.13.0 and vLLM 0.29.0 that the LLM experiments needed.

    python patches/apply_patches.py                  # the two patches every run needs
    python patches/apply_patches.py --entropy-bonus  # also the entropy-bonus ablation (Appendix G)

Each patch is an exact text replacement in the installed library. The script refuses to run if the
expected text is not found exactly once (another version is installed), keeps the original file
next to the patched one as <file>.orig, and does nothing if the patch is already applied.

1. TRL, trl/generation/vllm_generation.py. When training, TRL copies the trainer's weights into the
   vLLM sampler after every update. Gemma 4 has image and audio weights that the text-only vLLM model
   lacks, and the copy fails on them. The patch skips weights whose name mentions vision or audio and
   that vLLM reports it does not have. Every other weight is copied as before.
2. vLLM, vllm/lora/punica_wrapper/punica_gpu.py. vLLM's LoRA kernel requires its input to be stored
   contiguously in memory. gpt-oss-20b passes its attention layers a slice that is not, so evaluating
   a gpt-oss-20b adapter crashed. The patch makes a contiguous copy, which changes no values. It is the
   same change as vLLM pull request 36187. For the other three models the input is already contiguous.
3. (optional) TRL, trl/trainer/rloo_trainer.py. Adds an entropy bonus to the RLOO loss, for the
   entropy-bonus ablation: loss = RLOO loss - c * mean over answers of the summed next-token entropy,
   with c read from the environment variable ENTROPY_BONUS_COEF. Unset or 0 leaves the trainer unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

TRL_SYNC_OLD = """        elif self.mode == "colocate":
            for name, param in self._iter_named_params():
                self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, param)])
"""
TRL_SYNC_NEW = """        elif self.mode == "colocate":
            # Patched (DReST LLM experiments): skip Gemma 4's vision and audio weights, which the text-only vLLM model lacks.
            vllm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            skipped = getattr(self, "_skipped_non_text_param_names", None)
            for name, param in self._iter_named_params():
                if skipped is not None and name in skipped:
                    continue
                try:
                    vllm_model.load_weights([(name, param)])
                except ValueError as exc:
                    if "There is no module or parameter named" not in str(exc) or not any(
                        key in name for key in ("vision", "audio")
                    ):
                        raise
                    if skipped is None:
                        skipped = self._skipped_non_text_param_names = set()
                    skipped.add(name)
                    print(f"[patch] skipping non-text parameter absent from vLLM: {name}")
"""

VLLM_SHRINK_OLD = """        x = x.view(-1, x.shape[-1])
        lora_shrink(
"""
VLLM_SHRINK_NEW = """        x = x.view(-1, x.shape[-1]).contiguous()  # Patched (as vLLM PR 36187): gpt-oss passes a non-contiguous slice.
        lora_shrink(
"""

ENTROPY_OLD_1 = """            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
"""
ENTROPY_NEW_1 = """            if compute_entropy:
                if float(__import__("os").environ.get("ENTROPY_BONUS_COEF", "0") or 0) > 0 and torch.is_grad_enabled():
                    entropies = entropy_from_logits(logits)
                else:
                    with torch.no_grad():
                        entropies = entropy_from_logits(logits)
"""
ENTROPY_OLD_2 = """        per_sequence_loss = -torch.min(per_sequence_loss1, per_sequence_loss2)
        loss = per_sequence_loss.mean()
"""
ENTROPY_NEW_2 = ENTROPY_OLD_2 + """        _entropy_coef = float(__import__("os").environ.get("ENTROPY_BONUS_COEF", "0") or 0)
        if _entropy_coef > 0:
            _sequence_entropy = (entropies * completion_mask).sum(1)
            loss = loss - _entropy_coef * _sequence_entropy.mean()
            self._metrics["train" if self.model.training else "eval"]["entropy_bonus_seq"].append(_sequence_entropy.mean().item())
"""


def package_file(package: str, relative: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"{package} is not installed in this environment")
    return Path(list(spec.submodule_search_locations)[0]) / relative


def patch(path: Path, replacements, marker: str) -> None:
    text = path.read_text()
    if marker in text:
        print(f"already patched: {path}")
        return
    for old, new in replacements:
        if text.count(old) != 1:
            raise SystemExit(f"expected text not found exactly once in {path}: is a different version installed?")
        text = text.replace(old, new)
    backup = path.with_name(path.name + ".orig")
    if not backup.exists():
        shutil.copy(path, backup)
    path.write_text(text)
    print(f"patched: {path} (original kept as {backup.name})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entropy-bonus", action="store_true", help="also apply the entropy-bonus patch (ablation only)")
    args = ap.parse_args()
    patch(package_file("trl", "generation/vllm_generation.py"), [(TRL_SYNC_OLD, TRL_SYNC_NEW)], "_skipped_non_text_param_names")
    patch(package_file("vllm", "lora/punica_wrapper/punica_gpu.py"), [(VLLM_SHRINK_OLD, VLLM_SHRINK_NEW)], "as vLLM PR 36187")
    if args.entropy_bonus:
        patch(package_file("trl", "trainer/rloo_trainer.py"), [(ENTROPY_OLD_1, ENTROPY_NEW_1), (ENTROPY_OLD_2, ENTROPY_NEW_2)],
              "ENTROPY_BONUS_COEF")


if __name__ == "__main__":
    main()
