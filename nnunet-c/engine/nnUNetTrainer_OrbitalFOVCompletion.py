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

This module runs INSIDE nnunetv2 at train time (installed by the run script's
step 6b, alongside fov_completion_loader + fov_completion_planner). The lines that
touch nnU-Net internals (dataloader construction, train-key access) are marked
NOTE and mirror the audited OrbitalCascade.get_dataloaders; reconcile with your
installed nnU-Net version if the API differs.
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
        super().initialize()
        # FOV-safe prior augmentation defaults (only if the user hasn't overridden).
        os.environ.setdefault("CORRECTOR_DROP_ALL", "0.0")
        os.environ.setdefault("CORRECTOR_DROP_EACH", "0.10")
        self._fov_prior_note = (f"FOV prior aug: drop_all={os.environ['CORRECTOR_DROP_ALL']} "
                                f"drop_each={os.environ['CORRECTOR_DROP_EACH']} (jitter kept)")
        # snapshot cadence (§16.1): default 25 for the noisier completion landscape.
        self.save_every = int(os.environ.get("CORRECTOR_SAVE_EVERY", "25"))
        # completion manifest (case -> crop_type/severity/is_full_fov) for the planner.
        self._fov_manifest_path = os.environ.get(
            "CORRECTOR_FOV_MANIFEST",
            str(Path(self.preprocessed_dataset_folder) / "fov_completion_manifest.json"))
        self.print_to_log_file(f"[fov] {self._fov_prior_note}; save_every={self.save_every}; "
                               f"manifest={self._fov_manifest_path}")

    def _build_planner(self, dataset_tr) -> FOVCompletionBatchPlanner:
        records = _load_completion_records(self._fov_manifest_path)
        # NOTE: restrict the planner to THIS fold's train cases. dataset_tr exposes
        # its case keys; the accessor name varies by nnU-Net version.
        train_keys = None
        for attr in ("identifiers", "keys", "case_identifiers"):
            v = getattr(dataset_tr, attr, None)
            if callable(v):
                v = v()
            if v is not None:
                train_keys = set(v)
                break
        if train_keys is not None:
            records = [r for r in records if r["case_id"] in train_keys]
        case_index = FOVCaseIndex(records)
        # rank/worker-aware seed comes from the loader workers; seed the trainer-side
        # planner from nnU-Net's shuffle seed (workers re-seed per get_indices call).
        base_seed = int(getattr(self, "seed_for_shuffle", 12345) or 12345)
        return FOVCompletionBatchPlanner(
            case_index,
            full_fov_anchor_probability=float(os.environ.get("CORRECTOR_FOV_ANCHOR_FULL_PROB", "0.5")),
            base_seed=base_seed)

    def get_dataloaders(self):
        # If the FOV sampler is disabled, fall back to the cascade loaders.
        if os.environ.get("CORRECTOR_FOV_COMPLETION", "1") != "1":
            return super().get_dataloaders()

        # NOTE: this mirrors the audited OrbitalCascade.get_dataloaders construction,
        # swapping ONLY the train-loader class. Reconcile arg names with your nnU-Net.
        import importlib
        _tmod = importlib.import_module(
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer")

        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        initial_patch_size = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()[0] \
            if hasattr(self, "configure_rotation_dummyDA_mirroring_and_inital_patch_size") else patch_size
        tr_transforms = self.get_training_transforms(  # inherited (adds prior augs)
            patch_size, self.rotation_for_DA, self.deep_supervision_scales,
            self.mirror_axes, self.do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=True, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        val_transforms = self.get_validation_transforms(
            self.deep_supervision_scales, is_cascaded=True,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        planner = self._build_planner(dataset_tr)

        dl_tr = FOVCompletionStratifiedDataLoader(
            dataset_tr, self.batch_size, initial_patch_size, patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            planner=planner, preprocessed_folder=self.preprocessed_dataset_folder)
        dl_val = _tmod.nnUNetDataLoader(
            dataset_val, self.batch_size, patch_size, patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = _tmod.get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            return _tmod.SingleThreadedAugmenter(dl_tr, None), _tmod.SingleThreadedAugmenter(dl_val, None)
        return (_tmod.NonDetMultiThreadedAugmenter(dl_tr, None, allowed_num_processes,
                                                   max(1, allowed_num_processes // 2), None, pin_memory=self.device.type == "cuda"),
                _tmod.NonDetMultiThreadedAugmenter(dl_val, None, max(1, allowed_num_processes // 2),
                                                   max(1, allowed_num_processes // 2), None, pin_memory=self.device.type == "cuda"))
