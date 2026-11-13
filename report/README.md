# Report

The complete LaTeX source of the submitted report, ported from the team's Overleaf project.

- `main.tex`: the report.
- `references.bib`: its bibliography (BibTeX, `alpha` style).
- `figures/`, `tables/` and `numbers.txt`: generated from the committed results by `code/pipeline/make_report_assets.py` (readme §4.5). `main.tex` includes the figures and tables from these folders.

To build the PDF from this folder:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
