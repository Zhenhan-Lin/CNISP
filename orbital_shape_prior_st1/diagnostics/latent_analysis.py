"""
Latent space interpretability analysis.

Explores what the learned latent codes encode by correlating their
principal components with known anatomical metadata (volumes, centroids).

NOTE on interpretation:
    High correlation between latent PCs and anatomical metrics (e.g., globe
    volume, centroid position) does NOT necessarily indicate a problem.
    Different patients have genuinely different eye shapes and positions
    in the canonical patch space. A well-trained model SHOULD encode
    anatomically meaningful variation in its latent space.

    The diagnostics here help you understand WHAT the latent encodes,
    not whether it's "leaking" information. The actual test for whether
    alignment is sufficient is the centroid_shift analysis in
    reconstruction_qc.py (if unaligned dice << aligned dice, alignment
    needs improvement).
"""

from typing import Dict, List

import numpy as np


def analyze_latent_space(latents: np.ndarray, metadata: List[dict]) -> Dict:
    """
    PCA of latent vectors + correlation with metadata.

    Args:
        latents: [N, Z] trained or optimized latent vectors
        metadata: list of dicts from alignment metadata JSONs

    Returns:
        dict with:
            norms_mean/std: latent vector norm statistics
            pca_explained_variance_top5: variance captured by top 5 PCs
            correlations with metadata fields (if available)
    """
    norms = np.linalg.norm(latents, axis=1)

    # PCA
    centered = latents - latents.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained = (S ** 2) / np.sum(S ** 2)
    scores = U * S

    results = {
        "norms_mean": float(np.mean(norms)),
        "norms_std": float(np.std(norms)),
        "pca_explained_variance_top5": explained[:5].tolist(),
    }

    # Correlations with available metadata
    if not metadata or not metadata[0]:
        return results

    for field, label in [
        ("globe_volume_mm3", "globe_volume"),
        ("on_volume_mm3", "on_volume"),
    ]:
        if field in metadata[0]:
            vals = np.array([m.get(field, np.nan) for m in metadata])
            valid = ~np.isnan(vals)
            if valid.sum() > 5:
                for pc in range(min(3, scores.shape[1])):
                    corr = float(np.corrcoef(scores[valid, pc], vals[valid])[0, 1])
                    results[f"pc{pc}_vs_{label}_corr"] = corr

    if "globe_centroid_world" in metadata[0]:
        centroids = np.array([
            m.get("globe_centroid_world", [np.nan]*3) for m in metadata
        ])
        valid = ~np.isnan(centroids[:, 0])
        if valid.sum() > 5:
            for pc in range(min(3, scores.shape[1])):
                for ax, name in enumerate(["x", "y", "z"]):
                    corr = float(np.corrcoef(
                        scores[valid, pc], centroids[valid, ax]
                    )[0, 1])
                    results[f"pc{pc}_vs_centroid_{name}_corr"] = corr

    return results


def print_latent_report(analysis: Dict):
    print("\n" + "=" * 60)
    print("LATENT SPACE INTERPRETABILITY")
    print("=" * 60)

    print(f"Norms: {analysis['norms_mean']:.2f} ± {analysis['norms_std']:.2f}")
    print(f"PCA top-5 explained variance: "
          f"{[f'{v:.3f}' for v in analysis['pca_explained_variance_top5']]}")

    print("\nCorrelation with anatomical metrics:")
    print("  (High |corr| = latent encodes this variation; not necessarily bad)")
    for key, val in sorted(analysis.items()):
        if "corr" in key:
            strength = "strong" if abs(val) > 0.5 else "moderate" if abs(val) > 0.3 else "weak"
            print(f"  {key}: {val:+.3f} ({strength})")
    print("=" * 60)
