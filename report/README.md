# Report

The complete LaTeX source of the submitted report, ported from the team's Overleaf project.

- `main.tex`: the report.
- `references.bib`: its bibliography (BibTeX, `alpha` style).
- `figures/`, `tables/` and `numbers.txt`: generated from the committed results by `code/pipeline/make_report_assets.py`. `main.tex` includes the figures and tables from these folders.

## Building the PDF

This needs a LaTeX distribution with `pdflatex` and `bibtex` (TeX Live, MacTeX or MiKTeX), and no Python. The report uses only standard packages (`libertine`, `enumitem`, `booktabs`, `tabularx`, `xcolor`, `hyperref`, `fullpage`, `amsmath`), all included in a full TeX Live install. From the repository root:

```bash
cd report
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The paths in `main.tex` are relative to this folder, so run the commands from here. Alternatively, upload the `report/` folder to Overleaf and compile there.

## Regenerating the figures, tables and numbers

This uses the same Python environment as the rest of the repository: Python 3.12 with `code/requirements.txt` installed (readme §3). From the repository root, on macOS or Linux (for Windows or conda, see readme §3):

```bash
python3.12 -m venv .venv && source .venv/bin/activate      # skip if the environment exists
cd code
pip install -r requirements.txt                            # skip if already installed
python pipeline/make_report_assets.py                      # rewrites report/figures, report/tables and report/numbers.txt
git status --short ../report/tables ../report/numbers.txt  # empty = tables and numbers reproduced
python pipeline/verify_report_numbers.py                   # checks every number the report quotes; about 5-10 minutes
```

Run these on the committed results. If you ran `score.sh` first (readme §4.3), it re-measured the run times that the cost table's Time column reads, so restore them with `git checkout -- results` (from `code/`). Figure bytes can differ with the matplotlib version, fonts and operating system, but the plotted data does not (readme §4.5).
