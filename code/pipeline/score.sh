#!/usr/bin/env bash
# Score one or more settings from their committed run folders. Every step is deterministic (no LLM calls), so
# anyone can rerun this and get byte-identical outputs. Usage, from code/:
#     bash pipeline/score.sh abl_anyfit abl_informed_rules      # score
#     bash pipeline/score.sh --check abl_anyfit                  # score, then fail if any committed output changed
#
# Steps per setting (configs/<setting>.yaml):
#   1. tuned Best Fit retuned in the setting (no LLM, ~2.5 min per seed; finished seeds are skipped)
#   2. evaluate_final   held-out scores on the frozen test sets (#16)
#   3. check_integrity  integrity flags (#6)
#   4. diagnose         fresh_given_room, both packing modes (#5)
#   5. paired_stats     per-seed verdicts vs Best Fit and tuned Best Fit (#13; bootstrap uses a fixed seed)
#   6. summarise        tables and figures
set -euo pipefail
cd "$(dirname "$0")/.."

py="${PYTHON:-$(command -v python || command -v python3)}"   # override with PYTHON=/path/to/python
check=0
if [[ "${1:-}" == "--check" ]]; then check=1; shift; fi
[[ $# -gt 0 ]] || { echo "usage: bash pipeline/score.sh [--check] <setting> [<setting> ...]"; exit 2; }

for s in "$@"; do
    cfg="configs/$s.yaml"
    [[ -f "$cfg" ]] || { echo "no such config: $cfg"; exit 2; }
    echo "=== $s"
    "$py" pipeline/run_experiment.py  --config "$cfg" --methods tuned_bf
    "$py" pipeline/evaluate_final.py  --config "$cfg"
    "$py" pipeline/check_integrity.py --config "$cfg"
    "$py" pipeline/diagnose.py        --config "$cfg"
    "$py" pipeline/paired_stats.py    --config "$cfg"
    "$py" pipeline/summarise.py       --config "$cfg"
done

if [[ $check -eq 1 ]]; then
    dirs=(); for s in "$@"; do dirs+=("results/$s/"); done
    changed=$(git status --porcelain --untracked-files=all -- "${dirs[@]}")
    if [[ -n "$changed" ]]; then
        echo "MISMATCH: these outputs differ from the commit (M = changed, ?? = not committed):"; echo "$changed"; exit 1
    fi
    echo "OK: every committed output reproduced exactly"
fi
