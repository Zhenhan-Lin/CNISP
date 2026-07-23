#!/usr/bin/env bash
# One-shot: a built MASK_INDEX -> metrics_long.csv -> the 4 evaluation figures.
#
# Prereq: build the MASK_INDEX first (build_mask_index.py; see its --help / the
# eval section of the workflow). Then point this at that json:
#
#   bash simulation/evaluation/make_eval_figures.sh \
#       comparison/viz/evaluation__thick/mask_index.json
#
# Stages (all write under the mask_index's own directory):
#   1. build_metrics.py          -> metrics_long.csv   (the shared interface)
#   2. dice_quality_summary      -> Dice vs eff-res, 5-arm (replaces combined__thick)
#   3. surface_quality_summary   -> ASSD / HD95 / NSD figures
#   4. volume_agreement_summary  -> signed volume error across methods
#                                   (BLAND_ALTMAN=1 adds the Bland-Altman figure)
#   5. volume_stability_summary  -> volume CoV across steps
#   6. plausibility_summary      -> anatomical-plausibility figures
#   7. cross_resolution_summary  -> per-method cross-resolution Dice heatmaps
#
# RESUME by default: an existing metrics_long.csv (stage 1) and plausibility_long.csv
# (stage 6) are reused instead of recomputed -- a re-run only re-renders figures.
# Force rebuilds with REBUILD_METRICS=1 / PLAUS_ARGS="--recompute".
#
# Env overrides:
#   MODE=thick            sweep mode; MUST match the mask_index 'mode' field.
#   BA_STRUCTURE=Globe    structure for the Bland-Altman volume plot.
#   TAU_MM=<mm>           NSD tolerance (default = metrics.DEFAULT_TAU_MM).
#   COMMON_SAMPLES=1      restrict every figure to (case,step) present for ALL arms
#                         (fair like-for-like; off = each arm uses all it has).
#   PLAUS_ARGS="..."      extra flags forwarded verbatim to plausibility_summary.py
#                         (e.g. "--do-shape-reg --qualitative-case 10058_20330227
#                          --qualitative-step 5 --test-cases-map <map.json>").
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"            # simulation/evaluation
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

IDX="${1:?usage: make_eval_figures.sh <mask_index.json> [mode]}"
[[ -f "$IDX" ]] || { echo "[figs] mask_index not found: $IDX" >&2; exit 2; }
MODE="${2:-${MODE:-thick}}"
BA_STRUCTURE="${BA_STRUCTURE:-Globe}"
OUT="$(dirname "$IDX")"
CSV="$OUT/metrics_long.csv"
EVAL="simulation/evaluation"

COMMON=""; [[ "${COMMON_SAMPLES:-0}" == "1" ]] && COMMON="--common-samples"
TAU="";    [[ -n "${TAU_MM:-}" ]] && TAU="--tau-mm $TAU_MM"

echo "================================================================"
echo "[figs] index=$IDX"
echo "[figs] mode=$MODE  ba-structure=$BA_STRUCTURE  out=$OUT"
echo "================================================================"

# (1) metrics_long.csv -- RESUME by default: reuse an existing one so a re-run
# never recomputes the (slow) per-mask metrics. REBUILD_METRICS=1 forces a rebuild.
if [[ -f "$CSV" && "${REBUILD_METRICS:-0}" != "1" ]]; then
    echo "[figs] (1) reuse existing $CSV (set REBUILD_METRICS=1 to rebuild)"
else
    echo "[figs] (1) build_metrics -> $CSV"
    python3 "$EVAL/build_metrics.py" --mask-index "$IDX" --out-csv "$CSV" $TAU
fi

echo "[figs] (2) Dice vs effective resolution (5-arm; replaces combined__thick)"
python3 "$EVAL/dice_quality_summary.py"     --out "$OUT" --mode "$MODE" \
    --metrics-csv "$CSV" $COMMON $TAU ${DICE_ARGS:-}

echo "[figs] (3) surface quality (ASSD / HD95 / NSD)"
python3 "$EVAL/surface_quality_summary.py"  --out "$OUT" --mode "$MODE" \
    --metrics-csv "$CSV" $COMMON $TAU

BA_FLAG=""; [[ "${BLAND_ALTMAN:-0}" == "1" ]] && BA_FLAG="--bland-altman"
echo "[figs] (4) volume veracity (signed volume error${BA_FLAG:+ + Bland-Altman: $BA_STRUCTURE})"
python3 "$EVAL/volume_agreement_summary.py" --out "$OUT" --mode "$MODE" \
    --ba-structure "$BA_STRUCTURE" --metrics-csv "$CSV" $COMMON $TAU $BA_FLAG

echo "[figs] (5) volume stability (CoV across steps)"
python3 "$EVAL/volume_stability_summary.py" --out "$OUT" --mode "$MODE" \
    --metrics-csv "$CSV" $COMMON $TAU

# (6) plausibility -- auto-reuses an existing <out>/plausibility/plausibility_long.csv
# (the hours-long per-mask table); pass PLAUS_ARGS="--recompute" to force a rebuild.
echo "[figs] (6) plausibility (reuses plausibility_long.csv if present)"
python3 "$EVAL/plausibility_summary.py" --mask-index "$IDX" \
    --out "$OUT/plausibility" ${PLAUS_ARGS:-}

# (7) per-method cross-resolution Dice heatmaps (self-consistency across steps).
# Reads the MASK_INDEX directly (per-arm masks); one heatmap bundle per method +
# a by-method overview, under $OUT/cross_resolution/.
echo "[figs] (7) cross-resolution Dice heatmaps (per method)"
python3 "$EVAL/cross_resolution_summary.py" --mask-index "$IDX" --out "$OUT" \
    ${XRES_ARGS:-}

echo "================================================================"
echo "[figs] DONE -> figures under $OUT  (+ plausibility in $OUT/plausibility,"
echo "[figs]         cross-resolution heatmaps in $OUT/cross_resolution)"
echo "================================================================"
