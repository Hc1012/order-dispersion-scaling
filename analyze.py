"""Recompute every number in the README from the result JSONLs, and regenerate the figure.

    python analyze.py

Needs: numpy, matplotlib. Run inside the repo folder (the .jsonl files and this
script in the same directory). Reproduces the dispersion-law slopes, the
sign-correlation table, and the clipped-vs-exact 3B artifact comparison, then
rebuilds dispersion_scale_curve_final.png.
"""

import json
import os
import collections

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NS = [8, 16, 32, 48]

# model size -> candidate filenames (GitHub keeps the dots; local copies may use underscores)
MODELS = {
    0.5: ["full_0.5b.jsonl", "full_0_5b.jsonl"],
    1.5: ["full_1.5b.jsonl", "full_1_5b.jsonl"],
    3.0: ["full_3b_hp.jsonl"],   # exact log-odds run = the 3B of record
    7.0: ["full_7b.jsonl"],
}
CLIPPED_3B = ["full_3b.jsonl"]   # original probability run, for the artifact comparison
COL = {0.5: "#9bb8d3", 1.5: "#2b6cb0", 3.0: "#c05621", 7.0: "#1a7a4a"}
RNG = np.random.default_rng(0)


def find(names):
    for n in names:
        if os.path.exists(n):
            return n
    return None


def load(names):
    path = find(names)
    if path is None:
        raise FileNotFoundError(f"none of {names} found in {os.getcwd()}")
    return [json.loads(line) for line in open(path)]


def row_logits(r):
    """Per-ordering values on the logit scale.

    Exact (lp_YES - lp_NO) when 'logodds' is stored; otherwise the clipped
    logit of the stored probability. This is exactly why the clipped 3B run
    understates dispersion: identical 0.0/1.0 probabilities map to identical
    clipped logits and contribute zero spread.
    """
    if "logodds" in r:
        return np.asarray(r["logodds"], dtype=float)
    q = np.clip(np.asarray(r["q"], dtype=float), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def saturation_pct(rows):
    """Percent of stored per-ordering values pinned at the precision ceiling."""
    if "logodds" in rows[0]:
        vals = np.concatenate([np.abs(np.asarray(r["logodds"])) for r in rows])
        return float((vals > 13.8).mean() * 100.0)   # 13.8 ~= logit(1e-6), the old clip ceiling
    vals = np.concatenate([np.asarray(r["q"]) for r in rows])
    return float(((vals == 0.0) | (vals == 1.0)).mean() * 100.0)


def fit(rows, n_boot=1000):
    """Per-n pooled logit-MAD, OLS slope b vs ln n with bootstrap CI, and sign correlations."""
    row_mad = [float(np.abs(row_logits(r) - row_logits(r).mean()).mean()) for r in rows]
    idx_by_n = collections.defaultdict(list)
    for i, r in enumerate(rows):
        idx_by_n[r["n"]].append(i)

    pooled = {n: float(np.mean([row_mad[i] for i in idx_by_n[n]])) for n in NS}
    X = np.log(NS)
    b = float(np.polyfit(X, [pooled[n] for n in NS], 1)[0])

    boots = []
    for _ in range(n_boot):
        Y = [np.mean([row_mad[i] for i in RNG.choice(idx_by_n[n], len(idx_by_n[n]))]) for n in NS]
        boots.append(np.polyfit(X, np.asarray(Y), 1)[0])
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))

    sign = {}
    for n in NS:
        rs = [rows[i] for i in idx_by_n[n]]
        y = [1 if r["supported"] else 0 for r in rs]
        d = [float(row_logits(r).std()) for r in rs]
        sign[n] = float(np.corrcoef(d, y)[0, 1])

    return dict(pooled=pooled, b=b, lo=lo, hi=hi, sign=sign)


def main():
    results = {}

    print("=" * 64)
    print("VALIDATION")
    print("=" * 64)
    for sz, names in MODELS.items():
        rows = load(names)
        dupes = len(rows) - len({(r["qid"], r["n"], r["condition"]) for r in rows})
        au = auroc([r["supported"] for r in rows], [r["q_mean"] for r in rows])
        sat = saturation_pct(rows)
        kind = "exact log-odds" if "logodds" in rows[0] else "probability"
        print(f"  {sz:>4}B  rows={len(rows):<4} dupes={dupes}  AUROC={au:.4f}  "
              f"saturated={sat:5.1f}%  ({kind})")
        results[sz] = fit(rows)

    print()
    print("=" * 64)
    print("RESULT 1 — dispersion law (logit MAD per depth; slope b vs ln n)")
    print("=" * 64)
    print(f"  {'model':>6} " + " ".join(f"{'n='+str(n):>8}" for n in NS) + f"   {'b [95% CI]'}")
    for sz in MODELS:
        r = results[sz]
        bound = ">= " if sz == 7.0 else "   "
        print(f"  {sz:>5}B " + " ".join(f"{r['pooled'][n]:>8.3f}" for n in NS)
              + f"   {bound}{r['b']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")

    print()
    print("=" * 64)
    print("RESULT 2 — corr(per-claim dispersion, gold label)")
    print("=" * 64)
    print(f"  {'model':>6} " + " ".join(f"{'n='+str(n):>8}" for n in NS))
    for sz in MODELS:
        s = results[sz]["sign"]
        print(f"  {sz:>5}B " + " ".join(f"{s[n]:>+8.2f}" for n in NS))

    clipped_path = find(CLIPPED_3B)
    if clipped_path is not None:
        print()
        print("=" * 64)
        print("ARTIFACT CHECK — 3B: original (clipped) vs exact log-odds")
        print("=" * 64)
        crows = [json.loads(line) for line in open(clipped_path)]
        cfit = fit(crows)
        print(f"  clipped run : saturated={saturation_pct(crows):.1f}%  "
              f"b={cfit['b']:.3f} [{cfit['lo']:.3f}, {cfit['hi']:.3f}]")
        print(f"  exact run   : saturated={saturation_pct(load(MODELS[3.0])):.1f}%  "
              f"b={results[3.0]['b']:.3f} [{results[3.0]['lo']:.3f}, {results[3.0]['hi']:.3f}]")
        print("  -> storing probabilities at finite precision understates dispersion on")
        print("     confident models. Store log-odds (lp_YES - lp_NO) instead.")

    # ---- figure ----
    szs = sorted(results)
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

    for sz in szs:
        p = results[sz]["pooled"]
        ax[0].plot(NS, [p[n] for n in NS], "o-", color=COL[sz],
                   label=f"{sz}B" + (" (exact log-odds)" if sz == 3.0 else ""))
    ax[0].set_xscale("log"); ax[0].set_xticks(NS); ax[0].set_xticklabels(NS)
    ax[0].set_xlabel("evidence spans n"); ax[0].set_ylabel("order dispersion (logit MAD)")
    ax[0].legend(fontsize=9)
    ax[0].set_title("1. Dispersion grows with depth at every scale", fontsize=10)

    for sz in szs:
        r = results[sz]
        ax[1].errorbar([sz], [r["b"]], yerr=[[r["b"] - r["lo"]], [r["hi"] - r["b"]]],
                       fmt="o", color=COL[sz], capsize=4, ms=8)
    ax[1].plot(szs, [results[s]["b"] for s in szs], ":", color="gray", lw=1)
    if 7.0 in results:
        ax[1].annotate("15% saturated ->\nlower bound", (7.0, results[7.0]["b"]),
                       textcoords="offset points", xytext=(-84, 16), fontsize=8, color="#1a7a4a")
    ax[1].set_xscale("log"); ax[1].set_xticks(szs); ax[1].set_xticklabels([f"{s}B" for s in szs])
    ax[1].set_xlabel("model parameters"); ax[1].set_ylabel("dispersion-law slope b")
    ax[1].set_title("2. The slope of the law rises with scale", fontsize=10)

    for n in NS:
        ax[2].plot(szs, [results[s]["sign"][n] for s in szs], "o-", label=f"n={n}", alpha=0.85)
    ax[2].axhline(0, color="gray", lw=0.6)
    ax[2].set_xscale("log"); ax[2].set_xticks(szs); ax[2].set_xticklabels([f"{s}B" for s in szs])
    ax[2].set_xlabel("model parameters"); ax[2].set_ylabel("corr(dispersion, supported)")
    ax[2].legend(fontsize=9, title="spans")
    ax[2].set_title("3. Dispersion-as-signal strengthens with scale", fontsize=10)

    plt.tight_layout()
    plt.savefig("dispersion_scale_curve_final.png", dpi=200, bbox_inches="tight")
    print("\nfigure regenerated -> dispersion_scale_curve_final.png")


if __name__ == "__main__":
    main()
