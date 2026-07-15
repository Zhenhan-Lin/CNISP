#!/usr/bin/env python3
"""Route-A (native-cascade) wiring probe for the nnUNet-C corrector overhaul.

The training overhaul moves the CNISP prior OUT of the image (ch1..4) and into
nnUNet's **native cascade** slot: the prior becomes a per-case ``seg_prev`` that
nnUNet loads at runtime and ``MoveSegAsOneHotToDataTransform`` folds into the data
tensor *after* intensity augmentation. Turning that on ("Route A") means giving the
plan a cascade config (``previous_stage_name`` set). Three facts decide the exact
code for the preprocessing restructure + the predict-side rewire, and they can only
be read on a box with ``nnunetv2`` installed:

  Q1  num_input_channels + previous_stage_name fan-out
      - does ``determine_num_input_channels`` add len(foreground_labels) purely
        from ``configuration_manager.previous_stage_name`` (so we get 5 for free)?
      - does setting ``previous_stage_name`` on the 3d_fullres config force a REAL
        ``3d_lowres`` configuration to exist (else ``get_configuration`` KeyErrors
        during plan-parse / preprocess / train / predict)?

  Q2  prevseg storage contract
      - the exact ``folder_with_segs_from_previous_stage`` path the trainer builds,
      - the seg_prev filename + on-disk format ``load_case`` reads (``.b2nd`` blosc2
        vs ``.npz``/``.npy``; dtype; single integer channel), and
      - how nnUNet itself WRITES ``predicted_next_stage`` (so relocating a parallel
        "prior" dataset's ``{id}_seg.b2nd`` into that folder is format-valid).

  Q3  predict-side cascade consumption
      - how ``nnUNetv2_predict`` / ``nNNetPredictor`` ingests the previous-stage
        segmentation for a cascade config (CLI flag? auto-discovered folder?),
      - so ``run_corrector_predict.sh`` can feed a 1-ch CT + a CNISP prevseg folder
        instead of a 5-ch image.

Run on the GPU box and paste the whole stdout back:

    python nnunet-c/diagnostics/inspect_cascade_route.py

Optionally point it at a real preprocessed corrector dataset to list the on-disk
layout (data.b2nd / _seg.b2nd / predicted_next_stage/…):

    python nnunet-c/diagnostics/inspect_cascade_route.py \
        --preprocessed-dataset Dataset845_PHOTON_CT_CORR_C_cnisp \
        --configuration 3d_fullres --plan-name nnUNetPlansFinetune

Only depends on the Python stdlib + an importable ``nnunetv2`` (+ its deps). It
never trains, preprocesses, or writes anything; it only reads source + metadata.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import traceback
from pathlib import Path


# ── small print helpers ─────────────────────────────────────────────────────
def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"== {title}")
    print("=" * 78)


def sub(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 70 - len(title)))


def _safe(fn):
    """Run a probe section; never let one failure abort the rest of the report."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - this is a diagnostic; report everything
        print(f"\n[!!] section raised {type(e).__name__}: {e}")
        traceback.print_exc(limit=4)


def import_first(*candidates):
    """Import the first importable ``module:attr`` (or ``module``) from candidates.

    Returns (obj, dotted_name) or (None, None). ``module:attr`` fetches the attr;
    a bare ``module`` returns the module.
    """
    last = None
    for cand in candidates:
        try:
            if ":" in cand:
                mod_name, attr = cand.split(":", 1)
                mod = importlib.import_module(mod_name)
                return getattr(mod, attr), cand
            return importlib.import_module(cand), cand
        except Exception as e:  # noqa: BLE001
            last = f"{cand}: {type(e).__name__}: {e}"
    if last:
        print(f"    [not found] tried {list(candidates)} -> last error {last}")
    return None, None


def grep_source(obj, patterns, context_label=None):
    """Print every source line of ``obj`` matching any of ``patterns`` (substrings).

    ``obj`` may be a class, function, or module. Prints ``Lnn: <stripped line>``.
    Returns the number of matches (0 if source is unavailable).
    """
    try:
        src = inspect.getsource(obj)
    except (OSError, TypeError) as e:
        print(f"    [no source for {getattr(obj, '__name__', obj)}: {e}]")
        return 0
    if context_label:
        print(f"    (grepping {context_label} for {patterns})")
    n = 0
    for i, line in enumerate(src.splitlines(), 1):
        if any(p in line for p in patterns):
            print(f"    L{i}: {line.strip()}")
            n += 1
    if n == 0:
        print(f"    [no lines matched {patterns}]")
    return n


def print_source(obj, max_lines=200):
    """Print the full source of ``obj`` (capped)."""
    try:
        src = inspect.getsource(obj)
    except (OSError, TypeError) as e:
        print(f"    [no source for {getattr(obj, '__name__', obj)}: {e}]")
        return
    lines = src.splitlines()
    for line in lines[:max_lines]:
        print(f"    {line}")
    if len(lines) > max_lines:
        print(f"    ... [{len(lines) - max_lines} more lines truncated]")


# ── Q0: versions / environment ──────────────────────────────────────────────
def q0_versions():
    hr("Q0  versions + module locations")
    for name in ("nnunetv2", "batchgeneratorsv2", "blosc2", "numpy", "torch"):
        try:
            m = importlib.import_module(name)
            print(f"    {name:16s} {getattr(m, '__version__', '?'):12s} @ "
                  f"{getattr(m, '__file__', '?')}")
        except Exception as e:  # noqa: BLE001
            print(f"    {name:16s} [import failed: {type(e).__name__}: {e}]")
    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        print(f"    ${var} = {os.environ.get(var, '<unset>')}")


# ── Q1: num_input_channels + previous_stage_name fan-out ─────────────────────
def q1_num_input_channels():
    hr("Q1  num_input_channels + previous_stage_name fan-out")

    sub("determine_num_input_channels (does previous_stage_name alone -> +len(fg)?)")
    fn, name = import_first(
        "nnunetv2.utilities.label_handling.label_handling:determine_num_input_channels",
        "nnunetv2.utilities.label_handling:determine_num_input_channels",
    )
    if fn:
        print(f"    [{name}]")
        print_source(fn, max_lines=60)

    sub("ConfigurationManager.previous_stage_name (the trigger property)")
    cm, name = import_first(
        "nnunetv2.utilities.plans_handling.plans_handler:ConfigurationManager",
        "nnunetv2.utilities.plans_handling:ConfigurationManager",
    )
    if cm:
        print(f"    [{name}]")
        # the property + any next-stage/previous-stage bits
        grep_source(cm, ["previous_stage", "next_stage", "def "], context_label="ConfigurationManager")

    sub("PlansManager.get_configuration (does a missing config KeyError?)")
    pm, name = import_first(
        "nnunetv2.utilities.plans_handling.plans_handler:PlansManager",
        "nnunetv2.utilities.plans_handling:PlansManager",
    )
    if pm:
        print(f"    [{name}]")
        for meth in ("get_configuration", "available_configurations"):
            m = getattr(pm, meth, None)
            if m is not None:
                print(f"    # {meth}:")
                print_source(m, max_lines=40)

    sub("who calls get_configuration(previous_stage_name) in preprocess/plan?")
    for modname in (
        "nnunetv2.experiment_planning.experiment_planners.default_experiment_planner",
        "nnunetv2.preprocessing.preprocessors.default_preprocessor",
        "nnunetv2.experiment_planning.plan_and_preprocess_api",
    ):
        mod, _ = import_first(modname)
        if mod:
            print(f"    [{modname}]")
            grep_source(mod, ["previous_stage", "next_stage", "cascade",
                              "get_configuration"], context_label=modname)


# ── Q2: prevseg storage contract ────────────────────────────────────────────
def q2_prevseg_contract():
    hr("Q2  prevseg folder + load_case format + how nnUNet writes predicted_next_stage")

    sub("nnUNetTrainer: folder_with_segs_from_previous_stage + is_cascaded + predict_next_stage")
    tr, name = import_first(
        "nnunetv2.training.nnUNetTrainer.nnUNetTrainer:nnUNetTrainer",
    )
    if tr:
        print(f"    [{name}]")
        grep_source(
            tr,
            ["folder_with_segs_from_previous_stage", "is_cascaded",
             "previous_stage", "predicted_next_stage", "predict_next_stage"],
            context_label="nnUNetTrainer",
        )
        sub("nnUNetTrainer.predict_next_stage (how the prevseg .b2nd is produced/saved)")
        m = getattr(tr, "predict_next_stage", None)
        if m is not None:
            print_source(m, max_lines=120)

    sub("dataset load_case: seg_prev filename + format (blosc2/.npz/.npy, dtype, #channels)")
    for cand in (
        "nnunetv2.training.dataloading.nnunet_dataset:nnUNetDatasetBlosc2",
        "nnunetv2.training.dataloading.nnunet_dataset:nnUNetDatasetNumpy",
        "nnunetv2.training.dataloading.nnunet_dataset:nnUNetDataset",
    ):
        obj, nm = import_first(cand)
        if obj:
            print(f"    [{nm}]")
            # constructor (does it take folder_with_segs_from_previous_stage?) + load_case
            for meth in ("__init__", "load_case", "load_seg", "_load_seg_prev"):
                mm = getattr(obj, meth, None)
                if mm is not None and callable(mm):
                    print(f"    # {obj.__name__}.{meth}:")
                    print_source(mm, max_lines=80)

    sub("infer_dataset_class (which dataset class a preprocessed folder resolves to)")
    fn, nm = import_first(
        "nnunetv2.training.dataloading.utils:infer_dataset_class",
        "nnunetv2.training.dataloading.nnunet_dataset:infer_dataset_class",
    )
    if fn:
        print(f"    [{nm}]")
        print_source(fn, max_lines=40)

    sub("resample_and_save / export (the writer that fills predicted_next_stage)")
    for cand in (
        "nnunetv2.inference.export_prediction:resample_and_save",
        "nnunetv2.inference.export_prediction:export_prediction_from_logits",
    ):
        obj, nm = import_first(cand)
        if obj:
            print(f"    [{nm}]")
            print_source(obj, max_lines=70)


def q2b_on_disk(preprocessed_dataset, configuration, plan_name):
    hr("Q2b  on-disk preprocessed layout (optional; --preprocessed-dataset)")
    base = os.environ.get("nnUNet_preprocessed")
    if not base:
        print("    [$nnUNet_preprocessed unset -> skipping on-disk listing]")
        return
    if not preprocessed_dataset:
        print("    [no --preprocessed-dataset given -> skipping on-disk listing]")
        return
    ds = Path(base) / preprocessed_dataset
    print(f"    dataset dir: {ds}  exists={ds.is_dir()}")
    if not ds.is_dir():
        return
    print("    top-level entries:")
    for p in sorted(ds.iterdir()):
        tag = "/" if p.is_dir() else ""
        print(f"      {p.name}{tag}")
    data_dir = ds / f"{plan_name}_{configuration}"
    print(f"\n    data dir: {data_dir}  exists={data_dir.is_dir()}")
    if data_dir.is_dir():
        names = sorted(p.name for p in data_dir.iterdir())
        print(f"    first 12 of {len(names)} files: {names[:12]}")
    for cand in (ds / "predicted_next_stage", data_dir / "predicted_next_stage"):
        print(f"\n    predicted_next_stage probe: {cand}  exists={cand.is_dir()}")
        if cand.is_dir():
            for p in sorted(cand.iterdir()):
                print(f"      {p.name}{'/' if p.is_dir() else ''}")
                if p.is_dir():
                    inner = sorted(q.name for q in p.iterdir())
                    print(f"        first 8 of {len(inner)}: {inner[:8]}")


# ── Q3: predict-side cascade consumption ─────────────────────────────────────
def q3_predict_side():
    hr("Q3  predict-side cascade consumption (CLI flag + folder + one-hot injection)")

    sub("nNNetPredictor: is_cascaded / prev-stage handling in predict_from_raw_data")
    pred, name = import_first(
        "nnunetv2.inference.predict_from_raw_data:nnUNetPredictor",
    )
    if pred:
        print(f"    [{name}]")
        grep_source(
            pred,
            ["is_cascaded", "previous_stage", "prev_stage",
             "folder_with_segs_from_previous_stage", "add_segmentation_to_input",
             "num_seg_heads", "one_hot", "OneHot"],
            context_label="nnUNetPredictor",
        )
        for meth in ("predict_from_files", "_manage_input_and_output_lists",
                     "predict_logits_from_preprocessed_data",
                     "predict_from_data_iterator"):
            mm = getattr(pred, meth, None)
            if mm is not None:
                print(f"    # nnUNetPredictor.{meth} (grep prev-stage/onehot):")
                grep_source(mm, ["prev_stage", "previous_stage", "one_hot", "OneHot",
                                 "segs_from_prev", "add_segmentation_to_input"])

    sub("preprocessor used at predict-time: does it concat the prev-stage seg one-hot?")
    for cand in (
        "nnunetv2.preprocessing.preprocessors.default_preprocessor:DefaultPreprocessor",
    ):
        obj, nm = import_first(cand)
        if obj:
            print(f"    [{nm}]")
            for meth in ("run_case", "run_case_npy", "modality_dict"):
                mm = getattr(obj, meth, None)
                if mm is not None and callable(mm):
                    print(f"    # {obj.__name__}.{meth} (grep seg/onehot/prev):")
                    grep_source(mm, ["seg_prev", "previous_stage", "one_hot", "OneHot",
                                     "add_segmentation", "seg"])

    sub("predict_entry_point argparse: the CLI flag for previous-stage predictions")
    fn, nm = import_first(
        "nnunetv2.inference.predict_from_raw_data:predict_entry_point",
    )
    if fn:
        print(f"    [{nm}]")
        grep_source(fn, ["add_argument", "prev_stage", "previous"],
                    context_label="predict_entry_point")

    sub("MoveSegAsOneHotToDataTransform signature (channel it moves; #channels emitted)")
    for cand in (
        "batchgeneratorsv2.transforms.nnunet.seg_to_onehot:MoveSegAsOneHotToDataTransform",
        "batchgeneratorsv2.transforms.nnunet.seg_to_onehot:MoveSegAsOneHotToData",
    ):
        obj, nm = import_first(cand)
        if obj:
            print(f"    [{nm}]  __init__ signature:")
            try:
                print(f"      {inspect.signature(obj.__init__)}")
            except (TypeError, ValueError) as e:
                print(f"      [no signature: {e}]")
            grep_source(obj, ["source_channel", "all_labels", "def ",
                              "num_channels", "one_hot"], context_label=nm)
            break


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preprocessed-dataset", default=None,
                    help="e.g. Dataset845_PHOTON_CT_CORR_C_cnisp -> list on-disk layout")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--plan-name", default="nnUNetPlansFinetune")
    args = ap.parse_args()

    print("nnUNet-C Route-A cascade wiring probe")
    print(f"python: {sys.version.split()[0]}  cwd: {os.getcwd()}")

    _safe(q0_versions)
    _safe(q1_num_input_channels)
    _safe(q2_prevseg_contract)
    _safe(lambda: q2b_on_disk(args.preprocessed_dataset, args.configuration, args.plan_name))
    _safe(q3_predict_side)

    hr("DONE — paste the entire output above back into the chat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
