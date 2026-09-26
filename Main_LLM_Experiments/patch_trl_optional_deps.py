from __future__ import annotations

import site
from pathlib import Path


TARGET_SNIPPET = """with suppress_experimental_warning():
    from ..experimental.merge_model_callback import MergeModelCallback as _MergeModelCallback
    from ..experimental.winrate_callback import WinRateCallback as _WinRateCallback
"""

TARGET_WANDB_SNIPPET = """if is_wandb_available():
    import wandb
"""

OLD_REPLACEMENT_WANDB_SNIPPET = """if is_wandb_available():
    try:
        import wandb
    except Exception:
        wandb = None
else:
    wandb = None
"""

REPLACEMENT_WANDB_SNIPPET = """try:
    _trl_optional_wandb_available = is_wandb_available()
except Exception:
    _trl_optional_wandb_available = False
if _trl_optional_wandb_available:
    try:
        import wandb
    except Exception:
        wandb = None
else:
    wandb = None
"""

TARGET_WEAVE_SNIPPET = """if is_weave_available():
    import weave
    from weave import EvaluationLogger
    from weave.trace.context import weave_client_context
"""

REPLACEMENT_WEAVE_SNIPPET = """if is_weave_available():
    try:
        import weave
        from weave import EvaluationLogger
        from weave.trace.context import weave_client_context
    except Exception:
        weave = None
        EvaluationLogger = None

        class _DummyWeaveClientContext:
            @staticmethod
            def get_weave_client():
                return None

        weave_client_context = _DummyWeaveClientContext()
else:
    weave = None
    EvaluationLogger = None

    class _DummyWeaveClientContext:
        @staticmethod
        def get_weave_client():
            return None

    weave_client_context = _DummyWeaveClientContext()
"""

TARGET_WINRATE_IMPORT_SNIPPET = """    from ..experimental.winrate_callback import WinRateCallback as _WinRateCallback
"""

REPLACEMENT_WINRATE_IMPORT_SNIPPET = """    try:
        from ..experimental.winrate_callback import WinRateCallback as _WinRateCallback
    except Exception:
        class _WinRateCallback:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "WinRateCallback is unavailable because optional dependencies could not be imported."
                )
"""


REPLACEMENT_SNIPPET = """with suppress_experimental_warning():
    try:
        from ..experimental.merge_model_callback import MergeModelCallback as _MergeModelCallback
    except Exception:
        class _MergeModelCallback:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "MergeModelCallback is unavailable because optional mergekit dependencies could not be imported."
                )
    try:
        from ..experimental.winrate_callback import WinRateCallback as _WinRateCallback
    except Exception:
        class _WinRateCallback:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "WinRateCallback is unavailable because optional dependencies could not be imported."
                )
"""

TARGET_RLOO_VLLM_IMPORT_SNIPPET = """if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
"""

REPLACEMENT_RLOO_VLLM_IMPORT_SNIPPET = """if is_vllm_available():
    from vllm import LLM, SamplingParams
    try:
        from vllm.sampling_params import GuidedDecodingParams
        _TRL_VLLM_GUIDED_DECODING_KEY = "guided_decoding"
    except ImportError:
        from vllm.sampling_params import StructuredOutputsParams as GuidedDecodingParams
        _TRL_VLLM_GUIDED_DECODING_KEY = "structured_outputs"
"""

TARGET_RLOO_VLLM_KWARGS_SNIPPET = """                    "truncate_prompt_tokens": self.max_prompt_length,
                    "guided_decoding": guided_decoding,
                }
"""

REPLACEMENT_RLOO_VLLM_KWARGS_SNIPPET = """                    "truncate_prompt_tokens": self.max_prompt_length,
                }
                if guided_decoding is not None:
                    generation_kwargs[_TRL_VLLM_GUIDED_DECODING_KEY] = guided_decoding
"""

TARGET_RLOO_SAMPLING_PARAMS_SNIPPET = """                sampling_params = SamplingParams(**generation_kwargs)
"""

REPLACEMENT_RLOO_SAMPLING_PARAMS_SNIPPET = """                try:
                    sampling_params = SamplingParams(**generation_kwargs)
                except TypeError as exc:
                    if "truncate_prompt_tokens" not in str(exc):
                        raise
                    generation_kwargs.pop("truncate_prompt_tokens", None)
                    sampling_params = SamplingParams(**generation_kwargs)
"""

TARGET_VLLM_ASCEND_SNIPPET = """_vllm_ascend_available = _is_package_available("vllm_ascend")
"""

REPLACEMENT_VLLM_ASCEND_SNIPPET = """_vllm_ascend_available = _is_package_available("vllm_ascend")
if isinstance(_vllm_ascend_available, tuple):
    _vllm_ascend_available = bool(_vllm_ascend_available[0])
if _vllm_ascend_available:
    try:
        import importlib as _trl_safe_importlib
        _trl_safe_importlib.import_module("vllm_ascend")
    except Exception:
        _vllm_ascend_available = False
"""

TARGET_TRANSFORMERS_HUB_IMPORT_SNIPPET = """from huggingface_hub import (
    _CACHED_NO_EXIST,
    CommitOperationAdd,
    ModelCard,
    ModelCardData,
    constants,
    create_branch,
    create_commit,
    create_repo,
    hf_hub_download,
    hf_hub_url,
    is_offline_mode,
    list_repo_tree,
    snapshot_download,
    try_to_load_from_cache,
)
"""

REPLACEMENT_TRANSFORMERS_HUB_IMPORT_SNIPPET = """try:
    from huggingface_hub import (
        _CACHED_NO_EXIST,
        CommitOperationAdd,
        ModelCard,
        ModelCardData,
        constants,
        create_branch,
        create_commit,
        create_repo,
        hf_hub_download,
        hf_hub_url,
        is_offline_mode,
        list_repo_tree,
        snapshot_download,
        try_to_load_from_cache,
    )
except ImportError:
    from huggingface_hub import (
        _CACHED_NO_EXIST,
        CommitOperationAdd,
        ModelCard,
        ModelCardData,
        constants,
        create_branch,
        create_commit,
        create_repo,
        hf_hub_download,
        hf_hub_url,
        list_repo_tree,
        snapshot_download,
        try_to_load_from_cache,
    )

    def is_offline_mode():
        return bool(getattr(constants, "HF_HUB_OFFLINE", False))
"""


def patch_callbacks_file() -> Path:
    for root in site.getsitepackages():
        path = Path(root) / "trl" / "trainer" / "callbacks.py"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        changed = False
        replaced_merge_block = False
        if "optional mergekit dependencies could not be imported" not in text:
            if TARGET_SNIPPET not in text:
                raise RuntimeError(f"expected TRL merge import block not found in {path}")
            text = text.replace(TARGET_SNIPPET, REPLACEMENT_SNIPPET)
            changed = True
            replaced_merge_block = True
        if TARGET_WANDB_SNIPPET in text:
            text = text.replace(TARGET_WANDB_SNIPPET, REPLACEMENT_WANDB_SNIPPET)
            changed = True
        elif OLD_REPLACEMENT_WANDB_SNIPPET in text:
            text = text.replace(OLD_REPLACEMENT_WANDB_SNIPPET, REPLACEMENT_WANDB_SNIPPET)
            changed = True
        if TARGET_WEAVE_SNIPPET in text:
            text = text.replace(TARGET_WEAVE_SNIPPET, REPLACEMENT_WEAVE_SNIPPET)
            changed = True
        if (
            (not replaced_merge_block)
            and "WinRateCallback is unavailable because optional dependencies could not be imported." not in text
            and TARGET_WINRATE_IMPORT_SNIPPET in text
        ):
            text = text.replace(TARGET_WINRATE_IMPORT_SNIPPET, REPLACEMENT_WINRATE_IMPORT_SNIPPET)
            changed = True
        if changed:
            path.write_text(text, encoding="utf-8")
        return path
    raise RuntimeError("could not locate trl/trainer/callbacks.py in site-packages")


def patch_profiling_file() -> Path | None:
    for root in site.getsitepackages():
        path = Path(root) / "trl" / "extras" / "profiling.py"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        changed = False
        if TARGET_WANDB_SNIPPET in text:
            text = text.replace(TARGET_WANDB_SNIPPET, REPLACEMENT_WANDB_SNIPPET)
            changed = True
        elif OLD_REPLACEMENT_WANDB_SNIPPET in text:
            text = text.replace(OLD_REPLACEMENT_WANDB_SNIPPET, REPLACEMENT_WANDB_SNIPPET)
            changed = True
        if changed:
            path.write_text(text, encoding="utf-8")
            return path
        return None
    raise RuntimeError("could not locate trl/extras/profiling.py in site-packages")


def patch_rloo_file() -> Path | None:
    for root in site.getsitepackages():
        path = Path(root) / "trl" / "trainer" / "rloo_trainer.py"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        changed = False
        if "_TRL_VLLM_GUIDED_DECODING_KEY" not in text and TARGET_RLOO_VLLM_IMPORT_SNIPPET in text:
            text = text.replace(TARGET_RLOO_VLLM_IMPORT_SNIPPET, REPLACEMENT_RLOO_VLLM_IMPORT_SNIPPET)
            changed = True
        if "generation_kwargs[_TRL_VLLM_GUIDED_DECODING_KEY] = guided_decoding" not in text and TARGET_RLOO_VLLM_KWARGS_SNIPPET in text:
            text = text.replace(TARGET_RLOO_VLLM_KWARGS_SNIPPET, REPLACEMENT_RLOO_VLLM_KWARGS_SNIPPET)
            changed = True
        if "generation_kwargs.pop(\"truncate_prompt_tokens\", None)" not in text and TARGET_RLOO_SAMPLING_PARAMS_SNIPPET in text:
            text = text.replace(
                TARGET_RLOO_SAMPLING_PARAMS_SNIPPET,
                REPLACEMENT_RLOO_SAMPLING_PARAMS_SNIPPET,
            )
            changed = True
        if changed:
            path.write_text(text, encoding="utf-8")
            return path
        return None
    raise RuntimeError("could not locate trl/trainer/rloo_trainer.py in site-packages")


def patch_import_utils_file() -> Path | None:
    for root in site.getsitepackages():
        path = Path(root) / "trl" / "import_utils.py"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        changed = False
        if (
            "_trl_safe_importlib.import_module(\"vllm_ascend\")" not in text
            and TARGET_VLLM_ASCEND_SNIPPET in text
        ):
            text = text.replace(TARGET_VLLM_ASCEND_SNIPPET, REPLACEMENT_VLLM_ASCEND_SNIPPET)
            changed = True
        if "if isinstance(_vllm_ascend_available, tuple):" not in text:
            text = text.replace(
                """_vllm_ascend_available = _is_package_available("vllm_ascend")
if _vllm_ascend_available:
""",
                """_vllm_ascend_available = _is_package_available("vllm_ascend")
if isinstance(_vllm_ascend_available, tuple):
    _vllm_ascend_available = bool(_vllm_ascend_available[0])
if _vllm_ascend_available:
""",
            )
            changed = True
        if changed:
            path.write_text(text, encoding="utf-8")
            return path
        return None
    raise RuntimeError("could not locate trl/import_utils.py in site-packages")


def patch_transformers_hub_file() -> Path | None:
    for root in site.getsitepackages():
        path = Path(root) / "transformers" / "utils" / "hub.py"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        changed = False
        if "def is_offline_mode():" not in text and TARGET_TRANSFORMERS_HUB_IMPORT_SNIPPET in text:
            text = text.replace(
                TARGET_TRANSFORMERS_HUB_IMPORT_SNIPPET,
                REPLACEMENT_TRANSFORMERS_HUB_IMPORT_SNIPPET,
            )
            changed = True
        if changed:
            path.write_text(text, encoding="utf-8")
            return path
        return None
    raise RuntimeError("could not locate transformers/utils/hub.py in site-packages")


def main() -> None:
    path = patch_callbacks_file()
    print(f"patched {path}")
    profiling_path = patch_profiling_file()
    if profiling_path is not None:
        print(f"patched {profiling_path}")
    rloo_path = patch_rloo_file()
    if rloo_path is not None:
        print(f"patched {rloo_path}")
    import_utils_path = patch_import_utils_file()
    if import_utils_path is not None:
        print(f"patched {import_utils_path}")
    transformers_hub_path = patch_transformers_hub_file()
    if transformers_hub_path is not None:
        print(f"patched {transformers_hub_path}")


if __name__ == "__main__":
    main()
