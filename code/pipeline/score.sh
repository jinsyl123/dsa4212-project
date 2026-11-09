#!/usr/bin/env bash
# Score one or more settings from their committed run folders. Every step is deterministic (no LLM calls), so
# anyone can rerun this and get the same results. Usage, from code/:
#     bash pipeline/score.sh abl_anyfit abl_informed_rules      # score
#     bash pipeline/score.sh --check abl_anyfit                  # score, then fail if any committed result changed
#         (CSV row order, measured run times (time_s) and figure bytes may differ between runs; every score must match)
#     bash pipeline/score.sh --llm openrouter-gemini-3.5-flash-lite eoh_baseline
#         # --llm for configs whose results folder is named after the model (the baseline); same as the search
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
llm=()
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --check) check=1; shift ;;
        --llm) llm=(--llm "$2"); shift 2 ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
done
[[ $# -gt 0 ]] || { echo "usage: bash pipeline/score.sh [--check] [--llm NAME] <setting> [<setting> ...]"; exit 2; }

dirs=()
for s in "$@"; do
    cfg="configs/$s.yaml"
    [[ -f "$cfg" ]] || { echo "no such config: $cfg"; exit 2; }
    exp=$("$py" -c "import sys; sys.path.insert(0, 'pipeline'); from obpevo import config as C; \
print(C.load(sys.argv[1], llm=sys.argv[2] or None)['experiment'])" "$cfg" "${llm[1]:-}")
    dirs+=("results/$exp/")
    echo "=== $s -> results/$exp"
    "$py" pipeline/run_experiment.py  --config "$cfg" ${llm[@]+"${llm[@]}"} --methods tuned_bf
    "$py" pipeline/evaluate_final.py  --config "$cfg" ${llm[@]+"${llm[@]}"}
    "$py" pipeline/check_integrity.py --config "$cfg" ${llm[@]+"${llm[@]}"}
    "$py" pipeline/diagnose.py        --config "$cfg" ${llm[@]+"${llm[@]}"}
    "$py" pipeline/paired_stats.py    --config "$cfg" ${llm[@]+"${llm[@]}"}
    "$py" pipeline/summarise.py       --config "$cfg" ${llm[@]+"${llm[@]}"}
done

if [[ $check -eq 1 ]]; then
    "$py" pipeline/check_outputs.py "${dirs[@]}"   # CSV row order, run times and figure bytes may differ; nothing else
fi
