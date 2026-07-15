"""Orbital cascade corrector trainer (design Part 1: training overhaul).

Subclass of `nnUNetTrainer_corrector` (which only sets the finetune schedule) that
adds the three Part-1 training changes on top of nnUNet's **native cascade**
machinery. The CNISP prior plays the role of the "previous stage": the plan is
given a cascade config (`previous_stage_name`, done in build_finetune_plan.py),
so `self.is_cascaded` is True and nnUNet automatically (a) loads the per-case
CNISP prevseg `.b2nd` from `folder_with_segs_from_previous_stage`, (b) sets
`num_input_channels = 1 CT + 4 one-hot prior = 5`, (c) emits
`MoveSegAsOneHotToDataTransform` + the cascade morphological aug in the training
transforms, and (d) one-hots + vstacks the prior in whole-volume validation/predict.

What THIS trainer adds:
  1. `get_training_transforms` — insert the custom prior aug (centroid jitter +
     channel dropout, `corrector_augment.py`) after the stock cascade block and
     before deep-supervision downsampling. (Morphological aug §1.2.1 is the stock
     cascade block — reused, not duplicated.)
  2. `get_dataloaders` — swap the TRAIN loader for `StepStratifiednnUNetDataLoader`
     (one case per step stratum {3,6,9} + 1 bg per batch), with batch_size=4 and
     oversample=0.75. Behind env flag CORRECTOR_STRATIFIED (default on) → falls
     back to stock sampling when 0.
  3. `on_epoch_end` — snapshot `checkpoint_latest.pth` → `checkpoint_epoch_XXXX.pth`
     every `save_every` epochs so `select_checkpoint.py` can sweep them (nnUNet
     otherwise overwrites `checkpoint_latest`).

Hyperparameters read from env (exported by run_train.sh from corrector.yaml):
  CORRECTOR_JITTER_VOXELS = "z,y,x"  (default "4,2,2")
  CORRECTOR_DROP_ALL, CORRECTOR_DROP_EACH  (defaults 0.1, 0.25)
  CORRECTOR_STRATIFIED  (default "1")

Copied into nnunetv2 site-packages next to nnUNetTrainer_corrector.py +
corrector_augment.py + corrector_stratified_loader.py by run_train.sh /
run_corrector_predict.sh so `-tr nnUNetTrainer_OrbitalCascade` can discover it.
"""

from __future__ import annotations

import os
import shutil
from os.path import isfile, join

import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as _tmod
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_corrector import nnUNetTrainer_corrector
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import (
    DownsampleSegForDSTransform,
)

# installed alongside this file (heredoc); import from the same package
from nnunetv2.training.nnUNetTrainer.corrector_augment import (
    PriorCentroidJitterTransform,
    PriorChannelDropoutTransform,
)
from nnunetv2.training.nnUNetTrainer.corrector_stratified_loader import (
    StepStratifiednnUNetDataLoader,
)


def _env_floats(name, default):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


class nnUNetTrainer_OrbitalCascade(nnUNetTrainer_corrector):
    STRATA = (3, 6, 9)

    def initialize(self):
        super().initialize()  # schedule (epochs/lr) + network + label/config managers
        # ── stratification strata (env-driven) ──
        # Default = the thickness sweep {3,6,9}; the FOV-truncation experiment
        # overrides it with its truncation-level pseudo-steps via
        # CORRECTOR_STRATA="s1,s2,..." (the loader + batch_size follow self.STRATA).
        sv = os.environ.get("CORRECTOR_STRATA")
        if sv:
            self.STRATA = tuple(int(x) for x in sv.split(",") if x.strip())
        # ── prior-aug hyperparameters (env-driven) ──
        jv = os.environ.get("CORRECTOR_JITTER_VOXELS", "4,2,2")
        self._jitter_max = tuple(int(x) for x in jv.split(",") if x.strip())
        self._drop_all = _env_floats("CORRECTOR_DROP_ALL", 0.1)
        self._drop_each = _env_floats("CORRECTOR_DROP_EACH", 0.25)
        # ── stratified batching config ──
        self._stratified = os.environ.get("CORRECTOR_STRATIFIED", "1") == "1"
        if self._stratified:
            # one case per stratum + one background draw
            self.batch_size = 1 + len(self.STRATA)
            self.oversample_foreground_percent = len(self.STRATA) / self.batch_size

    # ── (1) training transforms: stock cascade chain + jitter/dropout ──
    def get_training_transforms(self, patch_size, rotation_for_DA, deep_supervision_scales,
                                mirror_axes, do_dummy_2d_data_aug, use_mask_for_norm=None,
                                is_cascaded=False, foreground_labels=None, regions=None,
                                ignore_label=None):
        assert is_cascaded, (
            "nnUNetTrainer_OrbitalCascade requires is_cascaded=True (set the plan's "
            "previous_stage_name in build_finetune_plan.py). Without it the CNISP prior "
            "is never moved into the data tensor."
        )
        compose = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm, is_cascaded, foreground_labels,
            regions, ignore_label,
        )
        # after MoveSegAsOneHot the prior occupies image channels 1..len(fg); insert
        # jitter+dropout just BEFORE deep-supervision downsampling (last transform).
        tfs = compose.transforms
        insert_at = next((i for i, t in enumerate(tfs)
                          if isinstance(t, DownsampleSegForDSTransform)), len(tfs))
        n_prior = len(foreground_labels) if foreground_labels else 4
        prior_idx = tuple(range(1, 1 + n_prior))
        tfs[insert_at:insert_at] = [
            PriorCentroidJitterTransform(prior_channel_indices=prior_idx,
                                         max_shift_voxels=self._jitter_max),
            PriorChannelDropoutTransform(prior_channel_indices=prior_idx,
                                         p_all=self._drop_all, p_each=self._drop_each),
        ]
        return compose

    # ── (2) dataloaders: stratified TRAIN loader (stock body, loader swapped) ──
    def get_dataloaders(self):
        if not self._stratified:
            return super().get_dataloaders()

        # symbols pulled from the stock trainer module's namespace (avoids guessing
        # import paths, and stays correct across nnUNet point releases).
        NonDetMultiThreadedAugmenter = _tmod.NonDetMultiThreadedAugmenter
        SingleThreadedAugmenter = _tmod.SingleThreadedAugmenter
        get_allowed_n_proc_DA = _tmod.get_allowed_n_proc_DA
        nnUNetDataLoader = _tmod.nnUNetDataLoader
        if self.dataset_class is None:
            self.dataset_class = _tmod.infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size,
         mirror_axes) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
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

        # TRAIN loader -> stratified subclass; VAL loader -> stock (whole-vol eval
        # in Phase 4 is the real val metric; per-epoch val stays a sanity check).
        dl_tr = StepStratifiednnUNetDataLoader(
            dataset_tr, self.batch_size, initial_patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling, strata=self.STRATA)
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

    # ── (3) periodic checkpoint snapshots for stratified selection (Phase 4) ──
    def on_epoch_end(self):
        super().on_epoch_end()
        every = getattr(self, "save_every", 50) or 50
        if (self.current_epoch + 1) % every == 0:
            src = join(self.output_folder, "checkpoint_latest.pth")
            if isfile(src):
                shutil.copyfile(
                    src, join(self.output_folder, f"checkpoint_epoch_{self.current_epoch + 1:04d}.pth"))
