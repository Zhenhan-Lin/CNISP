"""Tiny filesystem utilities shared by the input-staging scripts.

``safe_symlink`` is used by ``nnunet/prepare_inputs.py``,
``nnunet/prepare_smore_inputs.py`` and ``engine/build_smore_test_images.py``
to (re)point a channel-0 input symlink; ``top_level_root`` is the
container-bind-root helper used by the SMORE driver.
"""

from __future__ import annotations

from pathlib import Path


def safe_symlink(src: Path, dst: Path) -> None:
    """Point ``dst`` at ``src``, replacing any existing file/symlink."""
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)


def top_level_root(p: Path) -> Path:
    """Return ``/<first-path-component>`` (mirrors nnunetv2_build_datasets2)."""
    p = Path(p)
    try:
        parts = p.resolve().parts
    except Exception:  # noqa: BLE001
        parts = p.parts
    if len(parts) >= 2:
        return Path("/") / parts[1]
    return Path("/")
