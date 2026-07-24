"""
nnUNetTrainer_OrbitalFOVCompletion — FOV-completion finetune trainer.

Inheritance:  nnUNetTrainer_OrbitalCascade -> nnUNetTrainer_OrbitalFOVCompletion

It changes ONLY:
  1. the TRAIN loader -> FOVCompletionStratifiedDataLoader (condition x region
     stratified via FOVCompletionBatchPlanner); VAL loader stays stock;
  2. FOV-safe prior augmentation defaults (re-audit §6/§12.8): full-prior dropout
     0.0, per-channel dropout 0.10, centroid jitter retained — a missing-region
     patch with all prior channels dropped is an unlearnable target;
  3. snapshot cadence via CORRECTOR_SAVE_EVERY (default 25 for FOV; §16.1).

Everything else — cascade seg_prev handling, MoveSegAsOneHotToDataTransform,
per-channel resampling, optimizer/schedule, deep supervision, whole-volume
validation — is inherited from OrbitalCascade unchanged.

get_dataloaders() is a byte-for-byte copy of the proven
``nnUNetTrainer_OrbitalCascade.get_dataloaders`` body (audit §4): identical tuple
unpack, deep-supervision-scale accessor, is_cascaded flag, augmenter construction
and generator priming — the ONLY changes are the train-loader class
(FOVCompletionStratifiedDataLoader + planner) and the CORRECTOR_FOV_COMPLETION
fallback. Do not "clean it up"; drift from the parent is what §4 flagged.

RESUME caveat (audit §7, Option B): model/optimizer/scheduler resume is supported
via nnU-Net's checkpoint. The FOV SAMPLER stream (planner iteration/RNG, per-worker
state) is NOT persisted across a resume — worker-local RNG lives outside the trainer
process under multiprocessing. A resumed run continues training correctly but the
condition×region sampling sequence restarts; the long-run 50/30/20 + condition
balance is unaffected. For the first experiment (intended to run uninterrupted)
this is acceptable; exact sampler-stream resume is a follow-up.

This module runs INSIDE nnunetv2 at train time (installed by the run script's
step 6b, alongside fov_completion_loader + fov_completion_planner). The lines that
touch nnU-Net internals are marked NOTE and mirror the audited
OrbitalCascade.get_dataloaders; reconcile with your installed nnU-Net if the API differs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# When installed into nnunetv2.training.nnUNetTrainer.*, these are siblings.
try:
    from nnUNetTrainer_OrbitalCascade import nnUNetTrainer_OrbitalCascade
    from fov_completion_loader import FOVCompletionStratifiedDataLoader
    from fov_completion_planner import FOVCaseIndex, FOVCompletionBatchPlanner
except Exception:  # noqa: BLE001 -- allow import off the GPU box for inspection
    from engine.nnUNetTrainer_OrbitalCascade import nnUNetTrainer_OrbitalCascade  # type: ignore
    from engine.fov_completion_loader import FOVCompletionStratifiedDataLoader     # type: ignore
    from engine.fov_completion_planner import FOVCaseIndex, FOVCompletionBatchPlanner  # type: ignore


def _load_completion_records(path: str):
    man = json.loads(Path(path).read_text())
    return man["records"] if isinstance(man, dict) and "records" in man else man


class nnUNetTrainer_OrbitalFOVCompletion(nnUNetTrainer_OrbitalCascade):

    def initialize(self):
        # §8: FOV-safe prior-aug defaults MUST be set BEFORE super().initialize().
        # The parent OrbitalCascade.initialize() reads CORRECTOR_DROP_ALL/DROP_EACH
        # into self._drop_all/_drop_each during its own initialize(); setting them
        # afterwards would leave the parent's (0.10, 0.25) in effect on the direct
        # trainer path (the run script also exports them, but the class must be safe
        # standalone). setdefault -> a user/env override still wins.
        os.environ.setdefault("CORRECTOR_DROP_ALL", "0.0")
        os.environ.setdefault("CORRECTOR_DROP_EACH", "0.10")
        os.environ.setdefault("CORRECTOR_SAVE_EVERY", "25")
        super().initialize()
        self._fov_prior_note = (f"FOV prior aug: drop_all={self._drop_all} "
                                f"drop_each={self._drop_each} (jitter kept)")
        # snapshot cadence (§16.1): default 25 for the noisier completion landscape.
        self.save_every = int(os.environ["CORRECTOR_SAVE_EVERY"])
        # completion manifest (case -> crop_type/severity/is_full_fov) for the planner.
        self._fov_manifest_path = os.environ.get(
            "CORRECTOR_FOV_MANIFEST",
            str(Path(self.preprocessed_dataset_folder) / "fov_completion_manifest.json"))
        self.print_to_log_file(f"[fov] {self._fov_prior_note}; save_every={self.save_every}; "
                               f"manifest={self._fov_manifest_path}")

    def _build_planner(self, dataset_tr) -> FOVCompletionBatchPlanner:
        records = _load_completion_records(self._fov_manifest_path)
        # §9: obtain THIS fold's train identifiers and FAIL CLOSED if we can't. The
        # previous "if train_keys is not None" guard fell through to using ALL
        # manifest records when no accessor matched — that silently leaks validation
        # cases into the training sampler. dataset_tr's accessor name varies by
        # nnU-Net version; add yours here rather than letting it fail open.
        train_keys = None
        for attr in ("identifiers", "keys", "case_identifiers"):
            v = getattr(dataset_tr, attr, None)
            if callable(v):
                v = v()
            if v is not None:
                train_keys = set(v)
                break
        if train_keys is None:
            raise RuntimeError(
                "[fov] cannot obtain this fold's training identifiers from dataset_tr "
                "(tried identifiers/keys/case_identifiers). Refusing to fall back to all "
                "manifest records (that would leak validation cases). Add the correct "
                "accessor for your installed nnU-Net version.")
        manifest_keys = {r["case_id"] for r in records}
        missing = train_keys - manifest_keys
        if missing:
            raise RuntimeError(
                f"[fov] {len(missing)} fold-train case(s) are absent from the completion "
                f"manifest (e.g. {sorted(missing)[:5]}). The manifest case_id must equal "
                f"the nnU-Net dataset case id — supply a --case-map in the post-pass if the "
                f"dataset renamed them (report §6 item 3).")
        records = [r for r in records if r["case_id"] in train_keys]
        if not records:
            raise RuntimeError("[fov] no completion records match this fold's train keys.")
        case_index = FOVCaseIndex(records)
        base_seed = int(getattr(self, "seed_for_shuffle", 12345) or 12345)
        # §6: DDP rank separation (self.local_rank) so different ranks don't start from
        # the same stream. Per-worker separation is added later by the loader's
        # per-process reseed (fov_completion_loader._ensure_process_rng).
        global_rank = int(getattr(self, "local_rank", 0) or 0)
        return FOVCompletionBatchPlanner(
            case_index,
            full_fov_anchor_probability=float(os.environ.get("CORRECTOR_FOV_ANCHOR_FULL_PROB", "0.5")),
            base_seed=base_seed, global_rank=global_rank)

    def get_dataloaders(self):
        # If the FOV sampler is disabled, fall back to the cascade loaders.
        if os.environ.get("CORRECTOR_FOV_COMPLETION", "1") != "1":
            return super().get_dataloaders()

        # deferred import (this module stays importable off the GPU box for inspection;
        # get_dataloaders only ever runs inside nnU-Net at train time).
        import importlib
        _tmod = importlib.import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")

        # §4: BYTE-FOR-BYTE the proven nnUNetTrainer_OrbitalCascade.get_dataloaders
        # body — symbols pulled from the stock trainer module's namespace, correct
        # 4-tuple unpack, _get_deep_supervision_scales(), is_cascaded=self.is_cascaded,
        # named-kwarg augmenter construction, generator priming. The ONLY changes vs
        # the parent are: the TRAIN loader class (FOVCompletionStratifiedDataLoader +
        # planner/preprocessed_folder) and this CORRECTOR_FOV_COMPLETION gate.
        NonDetMultiThreadedAugmenter = _tmod.NonDetMultiThreadedAugmenter
        SingleThreadedAugmenter = _tmod.SingleThreadedAugmenter
        get_allowed_n_proc_DA = _tmod.get_allowed_n_proc_DA
        nnUNetDataLoader = _tmod.nnUNetDataLoader
        if self.dataset_class is None:
            self.dataset_class = _tmod.infer_dataset_class(self.preprocessed_dataset_folder)

        assert self.is_cascaded, (
            "nnUNetTrainer_OrbitalFOVCompletion requires is_cascaded=True (the plan's "
            "previous_stage_name, set by build_finetune_plan.py). Without it the CNISP "
            "prior is never moved into the data tensor.")

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size,
         mirror_axes) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(   # inherited (adds prior augs)
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        planner = self._build_planner(dataset_tr)

        # TRAIN loader -> FOV subclass (planner + region-aware center); VAL -> stock.
        dl_tr = FOVCompletionStratifiedDataLoader(
            dataset_tr, self.batch_size, initial_patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            planner=planner, preprocessed_folder=self.preprocessed_dataset_folder)
        dl_val = nnUNetDataLoader(
            dataset_val, self.batch_size, self.configuration_manager.patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None, num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2), seeds=None,
                pin_memory=self.device.type == "cuda", wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None, num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4), seeds=None,
                pin_memory=self.device.type == "cuda", wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val
