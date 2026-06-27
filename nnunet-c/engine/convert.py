"""The SINGLE nnUNet-C converter: CNISP/nnUNet prelabel mask -> nnUNet-C input.

This is the one place that turns an already-produced CNISP (or nnUNet) prelabel
into the corrector's network input. TRAIN and TEST both call ``convert_case``, so
the channel conversion is BYTE-IDENTICAL across them -- the only difference is
whether a GT label is also written (train) or not (test).

Fixed policy (never per-call, so train/test cannot drift):
  * 5 channels: ch0 = degraded CT (order 3), ch1..ch4 = binary prelabel (order 0)
  * structure order STRUCTS = [ON, Recti, Globe, Fat]   (ch1->ON .. ch4->Fat)
  * prelabel scheme = nnUNet {1,2,3,4}  (CNISP 032 and nnUNet preds both emit this)
  * reference grid = the source's ORIGINAL/dense (GT) grid; nnUNet later resamples
    that -> iso 0.5 plan via the per-channel resampler
  * GT label (TRAIN only) remapped to {1,2,3,4} (order 0)

nnUNet-C does NOT run CNISP here -- it only consumes CNISP's already-produced
masks. (CNISP inference + sweep live entirely in the CNISP layer / 032.)

Depends on lib.channels + lib.labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from lib import channels as _ch
from lib.labels import NNUNET_LABELS

# Fixed channel order: ch1->STRUCTS[0] .. chN->STRUCTS[-1]. DO NOT reorder; the
# trained corrector binds each channel index to this structure.
STRUCTS = ["ON", "Recti", "Globe", "Fat"]
N_CHANNELS = 5


def convert_case(
    case_id: str,
    ct_path,
    prelabel_path,
    ref_grid,
    experiment: str,
    images_dir: Path,
    *,
    gt_path=None,
    gt_struct_to_value: Optional[Dict[str, int]] = None,
    labels_dir: Optional[Path] = None,
    degraded_marker: Optional[str] = None,
) -> Dict:
    """Convert one (case, step) into nnUNet-C channels on ``ref_grid``.

    TRAIN: pass ``gt_path`` + ``labels_dir`` (+ ``gt_struct_to_value``, defaults
    to nnUNet {1,2,3,4}) -> writes _0000.._0004 + the label.
    TEST:  omit ``gt_path`` -> writes _0000.._0004 only (no label).

    ``degraded_marker`` pins ch0 to a degraded CT: None -> require the work_dir
    sparse marker ('/{experiment}/sparse_step_'); for the data/ train tree pass
    e.g. '/images/'.
    """
    if gt_path is not None:
        if labels_dir is None:
            raise ValueError("train conversion (gt_path set) requires labels_dir")
        return _ch.assemble_case(
            case_id=case_id, ct_path=Path(ct_path), gt_path=Path(gt_path),
            target_spacing=None, ref_grid=ref_grid,
            n_channels=N_CHANNELS, structures=STRUCTS,
            gt_struct_to_value=dict(gt_struct_to_value or NNUNET_LABELS),
            images_dir=images_dir, labels_dir=labels_dir, experiment=experiment,
            prelabel_path=Path(prelabel_path),
            prelabel_struct_to_value=dict(NNUNET_LABELS),
            degraded_marker=degraded_marker,
        )
    return _ch.assemble_inference_case(
        case_id=case_id, ct_path=Path(ct_path), target_spacing=None,
        ref_grid=ref_grid, n_channels=N_CHANNELS, structures=STRUCTS,
        images_dir=images_dir, experiment=experiment,
        prelabel_path=Path(prelabel_path),
        prelabel_struct_to_value=dict(NNUNET_LABELS),
        degraded_marker=degraded_marker,
    )
