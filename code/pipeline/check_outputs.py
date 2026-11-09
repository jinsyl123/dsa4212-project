"""Check that rescored outputs match the committed ones. Used by `score.sh --check`. Usage:
    python pipeline/check_outputs.py results/abl_anyfit/ [results/... ...]

Every changed or new file under the given folders is a mismatch, except a CSV whose only differences are
- row order (parallel workers finish in any order), or
- measured wall-clock time (TIME_COLUMNS), which no rerun can reproduce exactly.
Figures (FIGURE_SUFFIXES) are not checked: their bytes depend on the matplotlib version and fonts, and they are drawn
from the CSVs, which are. Every score, bin count and verdict must still match exactly. Exits 1 on any mismatch.
"""
import io
import subprocess
import sys

import pandas as pd

TIME_COLUMNS = {"time_s"}
FIGURE_SUFFIXES = (".png", ".pdf")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def same_csv(path: str) -> bool:
    old = pd.read_csv(io.StringIO(git("show", f"HEAD:./{path}")))
    new = pd.read_csv(path)
    if list(old.columns) != list(new.columns) or len(old) != len(new):
        return False
    cols = [c for c in old.columns if c not in TIME_COLUMNS]
    old, new = (d[cols].sort_values(cols).reset_index(drop=True) for d in (old, new))
    return old.equals(new)


def main(dirs: list[str]) -> int:
    changed = [l[3:] for l in git("status", "--porcelain", "--untracked-files=all", "--", *dirs).splitlines()]
    root = git("rev-parse", "--show-prefix").strip()                 # e.g. "code/": status paths are repo-relative
    bad = []
    for p in changed:
        rel = p[len(root):] if p.startswith(root) else p
        if rel.endswith(FIGURE_SUFFIXES):
            continue                                                  # figures differ across matplotlib versions
        status = git("status", "--porcelain", "--", rel)[:2]
        if status.strip() == "M" and rel.endswith(".csv") and same_csv(rel):
            continue                                                  # same content: only row order or timings moved
        bad.append(f"{status} {rel}")
    if bad:
        print("MISMATCH: these outputs differ from the commit (M = changed, ?? = not committed):")
        print("\n".join(bad))
        return 1
    print(f"OK: every committed output reproduced (CSV row order, {', '.join(sorted(TIME_COLUMNS))} and figures ignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
