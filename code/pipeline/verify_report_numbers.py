#!/usr/bin/env python3
"""Check every number the report quotes against the committed results. No LLM calls; about 5-10 minutes.

From code/:  python pipeline/verify_report_numbers.py [code dir]
Prints one line per group of checks (report section, the committed files it reads, checks run, failures),
then every failure, and exits with status 1 if any check fails.

Each check recomputes a value from the committed files and compares it with the value written in the report
(rounded as printed there). Nothing is read from the report's own tables or from make_report_assets.py, so this is
an independent second route to every number. Sources, all under code/:
  configs/instances/v0.yaml                      instance sets, regenerated deterministically
  results/<setting>/{eoh,randsample,tuned_bf}/seed<k>/best_program.py
                                                 the final programs: re-packed here (instance 0 of every set) and
                                                 re-scored under AnyFit for the empty-bins-withheld numbers
  results/<setting>/final_per_instance.csv       bins, L2 bound and excess of every program on every instance
  results/<setting>/paired_stats.csv             the pipeline's verdicts; recomputed here from final_per_instance.csv
                                                 (bootstrap + sign test) and required to agree on every row
  results/<setting>/diagnostics.csv              fresh-bin counts (phi) and AnyFit-mode excess
  results/<setting>/runs_index.csv               cost, tokens, valid share, unique programs, best evaluation
  results/<setting>/<method>/seed<k>/evals.csv   every evaluation of every run (checks runs_index.csv)
  results/<setting>/integrity.csv                repeat and shuffled rescoring of every best program
  results/<baseline>/tuned_bf/seed<k>/{candidates.jsonl,summary.json}
                                                 tuned Best Fit's sampled triples and fitted parameters
  results/abl_informed_rules/masked_choice.csv   report section 5.5, checks 1-2 (pipeline/masked_choice.py)
  results/abl_informed_rules/ratio_sweep.csv     report section 5.5, check 3 (pipeline/ratio_sweep.py)
Settings: Baseline = eoh_baseline__openrouter-gemini-3.5-flash-lite, A1a = abl_anyfit, A1b = abl_informed_rules,
Control = control_randsample, A2a = abl_mixed_shapes, A2b = abl_mixed_caps_informed.
"""
import sys, re, json
from math import comb
from pathlib import Path
import multiprocessing as mp
import numpy as np, pandas as pd, yaml
CODE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
RES = CODE / "results"
sys.path.insert(0, str(CODE / "pipeline"))
from obpevo.instances import generate_set
from obpevo.core import pack_funsearch, pack_anyfit, lower_bound, lower_bound_l2
from seed_counts import fisher_p                   # same two-sided Fisher exact test as the seed-count table
S = {"Baseline": "eoh_baseline__openrouter-gemini-3.5-flash-lite", "A1a": "abl_anyfit", "A1b": "abl_informed_rules", "Control": "control_randsample", "A2a": "abl_mixed_shapes", "A2b": "abl_mixed_caps_informed"}
SETS = ["test_" + x for x in ["id_5k", "size_1k", "size_10k", "cap200", "cap500", "cap500_scaled", "scale25", "scale65", "shape_k1.5", "shape_k5", "uniform"]]
cfg = yaml.safe_load(open(CODE / "configs/instances/v0.yaml"))
INST = {k: generate_set(k, cfg["sets"][k], cfg["seed"]) for k in SETS + ["test_control", "val"]}
FAILS, GROUPS = [], []
def group(section, sources):
    """Start a group of checks: the report section they cover and the committed files they read."""
    GROUPS.append(dict(section=section, sources=sources, n=0, failed=0))
def C(name, got, exp, tol=0.0051):
    """One check: got (recomputed from the committed files) against exp (as written in the report)."""
    g = GROUPS[-1]; g["n"] += 1
    ok = (got == exp) if isinstance(exp, (str, bool, list)) else abs(float(got) - float(exp)) <= tol
    if not ok: g["failed"] += 1; FAILS.append(f"[{g['section']}] {name}: recomputed {got}, report says {exp}")
def prog(s, label, seed): return RES / S[s] / label / f"seed{seed}" / "best_program.py"
def load(path):
    ns = {}; exec(path.read_text(), ns); return ns["priority"]
def pack(job):
    s, label, seed, st, k, mode = job
    i = INST[st][k]; f = load(prog(s, label, seed))
    return (*job, (pack_anyfit if mode == "anyfit" else pack_funsearch)(f, i["items"], i["capacity"]))
PI = {s: pd.read_csv(RES / S[s] / "final_per_instance.csv") for s in S}
DG = {s: pd.read_csv(RES / S[s] / "diagnostics.csv") for s in S}
RI = {s: pd.read_csv(RES / S[s] / "runs_index.csv") for s in S}
PS = {s: pd.read_csv(RES / S[s] / "paired_stats.csv") for s in S}
MODE = lambda s: "anyfit" if s == "A1a" else "funsearch"
EV = lambda s: "randsample" if s == "Control" else "eoh"

# 1. Re-pack every program on instance 0 of every set
jobs = [(s, lab, sd, st, 0, MODE(s)) for s in S for lab in (EV(s), "tuned_bf") for sd in range(10) for st in SETS]
group("All tables: final programs re-packed", "best_program.py, configs/instances/v0.yaml -> final_per_instance.csv")
if "fork" in mp.get_all_start_methods():                 # fork: workers must not re-run this script
    with mp.get_context("fork").Pool() as P: out = P.map(pack, jobs, chunksize=4)
else:                                                    # Windows has no fork: run serially (slower)
    out = [pack(j) for j in jobs]
bad = 0
for s, lab, sd, st, k, m, b in out:
    d = PI[s]; r = d[(d.label == lab) & (d.seed == sd) & (d.set == st) & (d.instance == INST[st][k]["id"])]
    C(f"repack {s} {lab} seed{sd} {st}", r.bins.tolist(), [b])
    if r.bins.tolist() != [b]: bad += 1
print(f"re-packed {len(out)} program/set pairs against final_per_instance.csv: {len(out)-bad} exact bin matches")
# 2a. raw arithmetic and second-pipeline agreement
group("All tables: raw arithmetic", "final_per_instance.csv, diagnostics.csv, runs_index.csv, evals.csv")
for s in S:
    d = PI[s]; d2 = d[d.set.isin(SETS)]
    C(f"{s} excess_l2 = (bins-lb_l2)/lb_l2", float(np.abs((d.bins - d.lb_l2) / d.lb_l2 - d.excess_l2).max()), 0, 1e-9)
    C(f"{s} all instances status ok", (d.status != "ok").sum(), 0)
    g = DG[s]; g = g[(g["mode"] == MODE(s)) & g.set.isin(SETS) & (g.label != "worst_fit")]  # Worst Fit is scored in its AnyFit form in final_per_instance
    m = d2.merge(g, on=["label", "seed", "set", "instance"], suffixes=("", "_d"))
    C(f"{s} diagnostics bins = final bins ({len(m)} rows)", (m.bins != m.bins_d).sum(), 0)
    for st in SETS:
        for k, i in enumerate(INST[st]):
            r = d[(d.label == "best_fit") & (d.set == st) & (d.instance == i["id"])]
            C(f"{s} {st} {k} L2 bound", int(r.lb_l2.iloc[0]), lower_bound_l2(i["items"], i["capacity"]), 0)
    # runs_index vs per-run evals.csv
    for _, r in RI[s][RI[s].kind == "evolved"].iterrows():
        e = pd.read_csv(RES / S[s] / EV(s) / f"seed{int(r.seed)}" / "evals.csv")
        C(f"{s} seed{r.seed} n_evals", len(e), r.n_evals, 0)
        C(f"{s} seed{r.seed} unique", e.code_hash.nunique(), r.n_unique_programs, 0)
        C(f"{s} seed{r.seed} ok", (e.status == "ok").sum(), r.n_ok, 0)
        ok = e[e.status == "ok"]; C(f"{s} seed{r.seed} best eval", int(ok[ok.train_excess == ok.train_excess.min()]["eval"].min()), r.best_eval, 0)  # earliest eval reaching the best score
# 2b. independent verdicts
def sign(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)
def verdict(d):
    if np.all(d == 0): return "tie"
    w, l = int((d < 0).sum()), int((d > 0).sum()); p = sign(w, l)
    m = d[np.random.default_rng(0).integers(0, len(d), size=(10000, len(d)))].mean(axis=1); lo, hi = np.percentile(m, [2.5, 97.5])
    return ("better" if d.mean() < 0 else "worse") if (hi < 0 or lo > 0) and p < .05 else "unclear"
group("All tables: verdicts", "final_per_instance.csv (recomputed) vs paired_stats.csv")
rows = []; mismatch = 0
for s in S:
    d = PI[s]; d = d[d.set.isin(SETS)]
    bf = d[d.label == "best_fit"].set_index(["set", "instance"]).excess_l2
    tu = d[d.label == "tuned_bf"].groupby(["set", "instance"]).excess_l2.mean()
    for (lab, sd, st), g in d[d.label.isin([EV(s), "tuned_bf"])].groupby(["label", "seed", "set"]):
        for ref, R in (("best_fit", bf), ("tuned_bf", tu)):
            if lab == "tuned_bf" and ref == "tuned_bf": continue
            x = g.set_index(["set", "instance"]).excess_l2; dd = (x - R.reindex(x.index)).values
            v = verdict(dd); rows.append(dict(s=s, label=lab, seed=sd, set=st, ref=ref, v=v, diff=dd.mean() * 100))
            t = PS[s]; q = t[(t.label == lab) & (t.seed == sd) & (t.set == st) & (t.reference == ref)]
            if q.verdict.iloc[0] != v: mismatch += 1
V = pd.DataFrame(rows); print(f"independent verdicts: {len(V)}, disagreements with paired_stats.csv: {mismatch}")
C("verdict disagreements", mismatch, 0, 0)

# 3. numbers quoted in the text
def n(s, st, v, ref="best_fit", label=None, seeds=None):
    q = V[(V.s == s) & (V.label == (label or EV(s))) & (V.set == "test_" + st) & (V.ref == ref) & (V.v == v)]
    if seeds is not None: q = q[q.seed.isin(seeds)]
    return len(q)
def D(s, st, seed=None, label=None):
    q = V[(V.s == s) & (V.label == (label or EV(s))) & (V.set == "test_" + st) & (V.ref == "best_fit")].set_index("seed")["diff"]
    return q if seed is None else q[seed]
def mean(s, st, label=None, seed=None):
    d = PI[s]; d = d[(d.label == (label or EV(s))) & (d.set == "test_" + st)]
    if seed is not None: d = d[d.seed == seed]
    return d.excess_l2.mean() * 100
def phi(s, st, seed=None, kind="evolved"):
    d = DG[s]; d = d[(d.kind == kind) & (d["mode"] == MODE(s)) & (d.set == st)]
    a = d.groupby("seed")[["n_fresh", "n_room"]].sum(); a = a.n_fresh / a.n_room.clip(lower=1)
    return a if seed is None else a[seed]
def af(s, label, seed, st):
    r = []
    for i in INST["test_" + st]:
        lb = lower_bound(i["items"], i["capacity"]); r.append((pack_anyfit(load(prog(s, label, seed)), i["items"], i["capacity"]) - lb) / lb)
    return np.mean(r) * 100
f = lambda a, b, c, d: fisher_p(a, b, c, d)
TR = dict(zip(S, ["train", "train", "train", "train", "train_mixed", "train_mixed_cap"]))
group("Abstract, 5.1: where evolved rules win and lose", "paired verdicts recomputed from final_per_instance.csv")
C("A1b better id_5k", n("A1b", "id_5k", "better"), 10, 0)
beat = sum(n(s, "id_5k", "better", "tuned_bf") for s in S); C("1 of 50 evolved rules beat their tuned Best Fit", beat, 1, 0)
C("Baseline better id_5k /10", n("Baseline", "id_5k", "better"), 5, 0); C("/6", n("Baseline", "id_5k", "better", seeds=range(6)), 3, 0)
C("seeds 4 gain", -D("Baseline", "id_5k", 4), 1.20); C("seed 5 gain", -D("Baseline", "id_5k", 5), 1.20)
for sd in (2, 8, 9): C(f"seed {sd} about 0.2", -D("Baseline", "id_5k", sd), .2, .05)
C("tuned better id_5k", n("Baseline", "id_5k", "better", label="tuned_bf"), 10, 0); C("tuned mean gain", -D("Baseline", "id_5k", label="tuned_bf").mean(), 2.48)
for st in ("id_5k", "size_1k", "size_10k", "uniform"): C(f"all worse than tuned {st}", n("Baseline", st, "worse", "tuned_bf"), 10, 0)
for st, x in (("cap500_scaled", 10), ("shape_k5", 10), ("shape_k1.5", 7)): C(f"tuned worse {st}", n("Baseline", st, "worse", label="tuned_bf"), x, 0)
C("tuned k5", mean("Baseline", "shape_k5", "tuned_bf"), 30.36); C("BF k5", mean("Baseline", "shape_k5", "best_fit"), 8.32)
for st, x in (("cap500_scaled", 10), ("scale25", 10), ("shape_k5", 10), ("shape_k1.5", 9)): C(f"evolved better than tuned {st}", n("Baseline", st, "better", "tuned_bf"), x, 0)
C("at most 2 worse on any set", max(n("Baseline", st.replace("test_", ""), "worse") for st in SETS), 2, 0)
for sd in (0, 1, 7):
    for st in ("id_5k", "shape_k1.5", "shape_k5"): C(f"seed {sd} tie {st}", V[(V.s == "Baseline") & (V.label == "eoh") & (V.seed == sd) & (V.set == "test_" + st) & (V.ref == "best_fit")].v.iloc[0], "tie")
group("5.2: what the rules do (fresh bins)", "diagnostics.csv, final_per_instance.csv, best_program.py re-scored under AnyFit")
C("fresh seeds", sorted(phi("Baseline", "train").loc[lambda x: x > .01].index.tolist()), [3, 4, 5])
d = PI["Baseline"]; a, b = [d[(d.label == "eoh") & (d.seed == k)].sort_values(["set", "instance"]).bins.values for k in (4, 5)]; C("seeds 4,5 identical bins", bool(np.array_equal(a, b)), True)
C("phi id seed4", phi("Baseline", "test_id_5k", 4), .26, .0051); C("phi k5 seed4", phi("Baseline", "test_shape_k5", 4), .45, .0051); C("k5 loss", D("Baseline", "shape_k5", 4), 2.49)
tp = phi("Baseline", "test_id_5k", kind="tuned").mean(); C("tuned phi id", tp, .33); C("tuned phi k5", phi("Baseline", "test_shape_k5", kind="tuned").mean(), .54)
C("seed 4 keeps", mean("Baseline", "id_5k", "best_fit") - af("Baseline", "eoh", 4, "id_5k"), .47)
C("tuned keeps", mean("Baseline", "id_5k", "best_fit") - np.mean([af("Baseline", "tuned_bf", k, "id_5k") for k in range(10)]), .16)
C("seed 4 k5 no empties", af("Baseline", "eoh", 4, "shape_k5"), 8.19); tk5 = np.mean([af("Baseline", "tuned_bf", k, "shape_k5") for k in range(10)]); C("tuned k5 no empties", tk5, 8.25)
C("seed 3 k1.5 with", mean("Baseline", "shape_k1.5", seed=3), 3.90, .02); C("seed 3 k1.5 without", af("Baseline", "eoh", 3, "shape_k1.5"), 3.99)
for s in ("Baseline", "A1b", "A2a", "A2b"):
    g = -D(s, "id_5k"); p = phi(s, "test_id_5k").reindex(g.index); C(f"{s}: gain>1 implies phi>0.1", int(((g > 1) & (p <= .1)).sum()), 0, 0)
group("5.3: Ablation 1 and the control", "final_per_instance.csv, diagnostics.csv, runs_index.csv, best_program.py")
C("A1a better", n("A1a", "id_5k", "better"), 6, 0); C("A1a max gain", (-D("A1a", "id_5k")).max(), .93)
C("A1a tuned", mean("A1a", "id_5k", "tuned_bf"), 3.17); C("A1a tuned gain", mean("A1a", "id_5k", "best_fit") - mean("A1a", "id_5k", "tuned_bf"), .91); C("A1a worse than tuned", n("A1a", "id_5k", "worse", "tuned_bf"), 8, 0)
d = DG["A1a"]; d = d[(d.kind == "evolved") & (d["mode"] == "funsearch") & (d.set == "test_id_5k") & d.seed.isin([1, 3, 6, 7])]
a = d.groupby("seed")[["n_fresh", "n_room"]].sum(); a = a.n_fresh / a.n_room; C("A1a phi min", a.min(), .64, .006); C("A1a phi max", a.max(), .73, .006)
e = d.groupby("seed").excess.mean() * 100; C("A1a excess min", e.min(), 60, .6); C("A1a excess max", e.max(), 82, .6)
C("p .036", f(10, 10, 3, 6), .036, .0006); C("p .033", f(10, 10, 5, 10), .033, .0006); C("A1b seed 1", mean("A1b", "id_5k", seed=1), 1.77); C("tuned", mean("A1b", "id_5k", "tuned_bf"), 1.60)
C("A1b fresh", int((phi("A1b", "train") > .01).sum()), 6, 0)
RX = re.compile(r"max\(bins\)|bins\.max\(\)|amax\(bins")
rd = {s: sorted(k for k in range(10) if RX.search(prog(s, EV(s), k).read_text())) for s in S}
C("A1b readers", rd["A1b"], [0, 2, 3, 4, 8, 9]); C("A2b readers", len(rd["A2b"]), 5, 0)
for s in ("Baseline", "A1a", "A2a"): C(f"{s} readers", rd[s], [])
col = lambda s, st: sorted(D(s, st).loc[lambda x: x > 100].index.tolist())
C("A1b cap500", col("A1b", "cap500"), [0, 2, 3, 4, 9]); C("A1b cap200", len(col("A1b", "cap200")), 4, 0); C("A1b scale25", len(col("A1b", "scale25")), 4, 0); C("A1b c500s none", col("A1b", "cap500_scaled"), [])
c = RI["Control"]; c = c[c.kind == "evolved"]; C("control unique", c.n_unique_programs.mean(), 30, .5); C("control eval 1", int((c.best_eval == 1).sum()), 10, 0); C("EoH unique", RI["Baseline"].query("kind=='evolved'").n_unique_programs.mean(), 207, .5)
for st in SETS: C(f"control = BF {st}", n("Control", st.replace("test_", ""), "tie"), 10, 0)
C("p fresh control 6", f(0, 10, 3, 6), .036, .0006); C("p fresh control 10", f(0, 10, 3, 10), .21, .005)
C("A1a seed 5 k5", D("A1a", "shape_k5", 5), .21); C("A1b median", RI["A1b"].query("kind=='evolved'").best_eval.median(), 217, 0); C("Base median", RI["Baseline"].query("kind=='evolved'").best_eval.median(), 180, 0); C("A1b k5 worse", n("A1b", "shape_k5", "worse"), 1, 0)
group("5.4: Ablation 2", "final_per_instance.csv, paired verdicts, Fisher tests")
C("A2a better", n("A2a", "id_5k", "better"), 9, 0); C("A2a gain", (-D("A2a", "id_5k")).mean(), .44); C("base gain", (-D("Baseline", "id_5k")).mean(), .30)
sw = lambda s, sd=None: V[(V.s == s) & (V.label == EV(s)) & (V.ref == "best_fit") & (V.v == "worse") & V.set.isin(["test_shape_k1.5", "test_shape_k5"]) & (V.seed.isin(sd if sd else range(10)))].seed.nunique()
C("A2a worse shape", sw("A2a"), 1, 0); C("A2a seed 6", D("A2a", "shape_k5", 6), .62); C("base 3/6", sw("Baseline", list(range(6))), 3, 0); C("base 3/10", sw("Baseline"), 3, 0); C("p .12", f(1, 10, 3, 6), .12, .005); C("p .58", f(1, 10, 3, 10), .58, .005)
C("A2a tuned k5", mean("A2a", "shape_k5", "tuned_bf"), 7.68); C("A2a tuned id", mean("A2a", "id_5k", "tuned_bf"), 2.63)
C("A2b cap200", col("A2b", "cap200"), []); C("A2b scale25", col("A2b", "scale25"), []); C("A2b cap500", col("A2b", "cap500"), [2, 8]); C("p .087", f(0, 10, 4, 10), .087, .0006); C("p .35", f(2, 10, 5, 10), .35, .005)
C("scale25 worse A1b", n("A1b", "scale25", "worse"), 5, 0); C("scale25 worse A2b", n("A2b", "scale25", "worse"), 0, 0); C("p .033", f(0, 10, 5, 10), .033, .0006); C("seed 4 sc25", D("A1b", "scale25", 4), .68)
C("A2b gain", (-D("A2b", "id_5k")).mean(), .35); C("A1b gain", (-D("A1b", "id_5k")).mean(), .56); C("A2b seed 8 cap500", D("A2b", "cap500", 8), 520, .5)
group("Table 10: validation-selected seed", "final_per_instance.csv, set val (lowest mean excess)")
for s, sd in (("A2b", 8), ("A1b", 1), ("A1a", 1), ("Baseline", 4), ("A2a", 6)):
    v = PI[s]; v = v[(v.label == EV(s)) & (v.set == "val")].groupby("seed").excess_l2.mean(); C(f"{s} val-selected", int(v.idxmin()), sd, 0)
group("5.5 and Table 9: failure attribution", "final_per_instance.csv, diagnostics.csv, best_program.py re-scored under AnyFit")
C("cap500 collapse value", mean("A1b", "cap500", seed=0), 1142, 1); C("cap500 no empties", af("A1b", "eoh", 0, "cap500"), .52)
C("seed 0 one bin per item", int(PI["A1b"].query("label=='eoh' and seed==0 and set=='test_cap500'").bins.min()), 5000, 0); C("phi cap500", phi("A1b", "test_cap500", 0), 1.00)
C("k5 attr", mean("Baseline", "shape_k5", seed=4), 10.81); C("tuned 5.19", mean("Baseline", "cap500_scaled", "tuned_bf"), 5.19); C("tuned 5.20", mean("A1a", "cap500_scaled", "tuned_bf"), 5.20)
C("A1b any cap500 no empties", max(af("A1b", "eoh", k, "cap500") for k in (0, 2, 3, 4, 9)), .5, .03); C("A2b seed2 cap500 with", mean("A2b", "cap500", seed=2), 223, .6); C("A2b seed8 with", mean("A2b", "cap500", seed=8), 520, .6); C("A2b no empties", max(af("A2b", "eoh", k, "cap500") for k in (2, 8)), .5, .03)
group("5.6 and Table 10: cost and integrity", "runs_index.csv, integrity.csv")
costs = {s: RI[s].query("kind=='evolved'").cost_usd.mean() for s in S}; C("cost lo", min(v for k, v in costs.items() if k != "Control"), .17); C("cost hi", max(costs.values()), .23); C("control cost", costs["Control"], .09)
vr = [(RI[s].query("kind=='evolved'").n_ok / RI[s].query("kind=='evolved'").n_evals).mean() for s in S]; C("about 99% valid", min(vr), .99, .005)
for s in S:
    t = pd.read_csv(RES / S[s] / "integrity.csv"); C(f"{s} integrity", int(t.repeat_mismatch.fillna(0).sum() + t.shuffle_mismatch.fillna(0).sum()), 0, 0)
group("6: discussion ranges", "final_per_instance.csv")
ks = lambda s: PI[s].query("label=='tuned_bf' and set=='test_shape_k5'").groupby("seed").excess_l2.mean() * 100
C("tuned k5 min", ks("Baseline").min(), 27.7, .051); C("max", ks("Baseline").max(), 35.2, .051); C("mixed min", ks("A2a").min(), 6.9, .051); C("mixed max", ks("A2a").max(), 9.8, .051)
group("3-4: instance sets and bounds", "configs/instances/v0.yaml (regenerated)")
big = np.mean([(i["items"] > 50).mean() for i in INST["test_scale65"]]); C("63% of scale65 exceed half", big, .63, .006)
diffb = {st: sum(lower_bound(i["items"], i["capacity"]) != lower_bound_l2(i["items"], i["capacity"]) for i in INST[st]) for st in SETS + ["test_control"]}
C("L1/L2 differ only on scale65, control", sorted(k for k, v in diffb.items() if v), ["test_control", "test_scale65"])
group("5.1: sets where evolved and tuned differ < 0.1 pp", "final_per_instance.csv")
for st in ("cap200", "cap500", "scale65"):
    C(f"evolved vs tuned within 0.1pp on {st}", abs(PI["Baseline"].query(f"label=='eoh' and set=='test_{st}'").excess_l2.mean() - PI["Baseline"].query(f"label=='tuned_bf' and set=='test_{st}'").excess_l2.mean()) * 100 < .1, True)
group("4.3, 5.2: tuned Best Fit search and fitted parameters", "tuned_bf/seed<k>/candidates.jsonl, summary.json")
# 4. tuned Best Fit's search space, pooled search and fitted parameters (sections 4.3 and 5.2)
from obpevo import tuned_bf as TB
TBD = sorted((RES / S["Baseline"] / "tuned_bf").glob("seed*"))
C("tuned seeds", len(TBD), 10, 0); C("space size", TB.space_size(), 37400, 0); C("one seed covers %", 299 / TB.space_size() * 100, 0.8, 0.051)
pooled = {tuple(sorted(json.loads(l)["params"].items())) for d in TBD for l in open(d / "candidates.jsonl")} - {tuple(sorted(TB.BEST_FIT.items()))}
C("pooled distinct triples", len(pooled), 2881, 0)
BP = [json.load(open(d / "summary.json")) for d in TBD]; bx = [b["best_train_excess"] * 100 for b in BP]
C("pooled best", min(bx), 2.36); C("median seed", np.median(bx), 2.41); C("pooled gain", np.median(bx) - min(bx), 0.05)
pr = lambda k: [b["best_params"][k] for b in BP]
C("lo range", [min(pr("lo")), max(pr("lo"))], [3, 4]); C("hi range", [min(pr("hi")), max(pr("hi"))], [29, 34]); C("penalty range", [min(pr("penalty")), max(pr("penalty"))], [64, 97])
C("hi share at C=500 below 7%", max(pr("hi")) / 500 * 100 < 7, True); C("hi share at C=100", max(pr("hi")), 34, 0)

# 5. Report section 5.5, checks 1-3, from pipeline/masked_choice.py and pipeline/ratio_sweep.py
group("5.5: AnyFit and masked choice (checks 1-2)", "abl_informed_rules/masked_choice.csv")
mc = pd.read_csv(RES / S["A1b"] / "masked_choice.csv"); mcv = lambda k, st, m: mc.query("seed == @k and set == @st and mode == @m").delta_pp.iloc[0]
coll = [(k, st) for k in (0, 2, 3, 4, 9) for st in ("test_cap200", "test_cap500", "test_scale25") if mcv(k, st, "funsearch") > 100]
C("withheld: collapsed sets within 0.03 pp of Best Fit", max(abs(mcv(k, st, "anyfit")) for k, st in coll) <= .03, True)
C("masked: seeds 0, 2, 3, 9 match Best Fit to 2 dp", max(abs(mcv(k, st, "masked")) for k in (0, 2, 3, 9) for st in ("test_id_5k", "test_cap200", "test_cap500", "test_scale25")) < .005, True)
C("seed 4 keeps 0.81 pp masked", -mcv(4, "test_id_5k", "masked"), .81); C("seed 4 loses 0.21 pp AnyFit", mcv(4, "test_id_5k", "anyfit"), .21)
group("5.5: ratio sweep (check 3)", "abl_informed_rules/ratio_sweep.csv")
rs = pd.read_csv(RES / S["A1b"] / "ratio_sweep.csv"); rsv = lambda k, s: rs.query("seed == @k and set == @s").delta_pp.iloc[0]
import ratio_sweep as RS
C("five instances of 2,000 items per ratio", [RS.N_INST, RS.N_ITEMS], [5, 2000])
C("ratios 0.36 to 0.13, plus capacity 1,000 at 0.40", sorted(rs.rho_target.unique().round(2).tolist()), [.13, .18, .23, .27, .31, .36, .4])
for r, x, tol in ((.36, -.3, .051), (.31, 4.7, .051), (.27, 67, .51), (.23, 201, .51)): C(f"seed 0 at rho {r}", rsv(0, f"sweep_rho{r:.2f}"), x, tol)
C("seed 0 collapse (> 100 pp) between 0.27 and 0.23", [rsv(0, "sweep_rho0.27") > 100, rsv(0, "sweep_rho0.23") > 100], [False, True])
for k in (2, 3, 9): C(f"seed {k} same pattern", [rsv(k, "sweep_rho0.36") < 0, rsv(k, "sweep_rho0.31") > 0, rsv(k, "sweep_rho0.23") > 100], [True, True, True])
C("seed 4 at 0.18", rsv(4, "sweep_rho0.18"), 18, .51); C("seed 4 at 0.13", rsv(4, "sweep_rho0.13"), 290, .51)
C("capacity 1,000: no collapse", rs.query("set == 'sweep_cap1000'").delta_pp.max() < 100, True); C("seed 2 at capacity 1,000", rsv(2, "sweep_cap1000"), 42, .51)

print(f"{'report section':58} {'checks':>6} {'failed':>6}  sources")
for g in GROUPS: print(f"{g['section'][:58]:58} {g['n']:>6} {g['failed']:>6}  {g['sources']}")
print(f"checks run: {sum(g['n'] for g in GROUPS)}, failed: {len(FAILS)}"); print("\n".join(FAILS) or "no failures")
sys.exit(1 if FAILS else 0)
