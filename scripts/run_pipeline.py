#!/usr/bin/env python3
"""Full pipeline: cleaning -> RFM -> clustering comparison -> segments -> CLV.

Outputs (in results/): metric tables, segment summary, customer table used by the
Streamlit app, and figures.
Usage: python scripts/run_pipeline.py
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.clustering import compare, fit_predict, METHODS, scores, stability  # noqa: E402
from segmentation.clv import holdout_validation, predict_clv, summary  # noqa: E402
from segmentation.data import clean, load_raw  # noqa: E402
from segmentation.rfm import build_rfm, scale_rfm  # noqa: E402
from segmentation.segments import name_clusters, profile, segment_table  # noqa: E402

RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
K = 4
FINAL_METHOD = "K-means"
CALIBRATION_END = pd.Timestamp("2011-09-09")
HORIZON_DAYS = 180
COLORS = {"Champions": "#2a9d8f", "Loyal customers": "#457b9d",
          "Promising newcomers": "#e9c46a", "Hibernating": "#e76f51"}


def main():
    FIG.mkdir(parents=True, exist_ok=True)

    # 1. Data
    tx, cleaning = clean(load_raw())
    end = tx["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    print("Cleaning:", cleaning)

    # 2. RFM
    rfm = build_rfm(tx, snapshot=end)
    X, _ = scale_rfm(rfm)

    # 3. Method comparison
    comp = compare(X)
    comp.to_csv(RESULTS / "clustering_comparison.csv", index=False)
    at_k = comp[comp["k"] == K].set_index("method")
    stab = {m: stability(X, m, K) for m in METHODS}
    at_k["stability_ari"] = [stab[m][0] for m in at_k.index]
    at_k.round(3).to_csv(RESULTS / f"comparison_k{K}.csv")
    print(at_k.round(3))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for m in METHODS:
        c = comp[comp["method"] == m]
        axes[0].plot(c["k"], c["silhouette"], "o-", label=m)
        axes[1].plot(c["k"], c["davies_bouldin"], "o-", label=m)
    for ax, title in zip(axes, ["Silhouette (higher is better)",
                                "Davies-Bouldin (lower is better)"]):
        ax.axvline(K, color="0.6", ls="--", lw=1)
        ax.set_xlabel("number of clusters k")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "method_comparison.png", dpi=120)
    plt.close(fig)

    # 4. Final segments
    labels = fit_predict(FINAL_METHOD, X, K)
    prof = profile(rfm, labels)
    names = name_clusters(prof)
    rfm["Segment"] = pd.Series(labels, index=rfm.index).map(names)

    # 5. CLV: holdout validation, then fit on the full period
    validation, val_df = holdout_validation(tx, CALIBRATION_END, end)
    (RESULTS / "clv_validation.json").write_text(json.dumps(validation, indent=2))
    print("CLV validation:", {k: round(v, 3) for k, v in validation.items()})

    clv, bg, gg = predict_clv(tx, end, HORIZON_DAYS)
    # P(alive) is not used: on one year of data the dropout parameter a is ~0 (as with
    # the lifetimes package), so churn is not identified and P(alive) is ~1 for everyone.
    rfm = rfm.join(clv[["pred_purchases", "clv"]])
    rfm.rename(columns={"clv": f"clv_{HORIZON_DAYS}d"}, inplace=True)

    clv_by_seg = (rfm.groupby("Segment")
                     .agg(clv_total=(f"clv_{HORIZON_DAYS}d", "sum"),
                          clv_per_customer=(f"clv_{HORIZON_DAYS}d", "mean")))
    table = segment_table(prof, names, clv_by_seg)
    table.round(3).to_csv(RESULTS / "segments.csv")
    print(table.drop(columns="action").round(2).to_string())

    params = {"bg_nbd": dict(r=bg.r, alpha=bg.alpha, a=bg.a, b=bg.b),
              "gamma_gamma": dict(p=gg.p, q=gg.q, v=gg.v),
              "final_scores": scores(X, labels), "cleaning": cleaning,
              "k": K, "method": FINAL_METHOD, "horizon_days": HORIZON_DAYS}
    (RESULTS / "model_summary.json").write_text(json.dumps(params, indent=2, default=float))

    out = rfm.reset_index()[["CustomerID", "Country", "Recency", "Frequency", "Monetary",
                             "Segment", "pred_purchases", f"clv_{HORIZON_DAYS}d"]]
    out.round(3).to_csv(RESULTS / "customers.csv", index=False)

    # Figures: segments
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    order = list(table.index)
    for ax, col, log in zip(axes, ["Recency", "Frequency", "Monetary"], [False, True, True]):
        data = [rfm.loc[rfm["Segment"] == s, col] for s in order]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True)
        for patch, s in zip(bp["boxes"], order):
            patch.set_facecolor(COLORS[s])
        ax.set_xticks(range(1, len(order) + 1), [s.replace(" ", "\n") for s in order], fontsize=8)
        ax.set_title(col + (" (log scale)" if log else " (days)"))
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "segment_profiles.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(order))
    ax.bar(x - 0.2, table["customer_share"] * 100, 0.4, label="% of customers", color="0.7")
    ax.bar(x + 0.2, table["revenue_share"] * 100, 0.4, label="% of revenue",
           color=[COLORS[s] for s in order])
    ax.set_xticks(x, order)
    ax.set_ylabel("%")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Customers vs revenue by segment")
    fig.tight_layout()
    fig.savefig(FIG / "segment_value.png", dpi=120)
    plt.close(fig)

    # Figure: CLV validation (mean predicted vs actual by number of calibration purchases)
    cal = summary(tx, CALIBRATION_END)
    grp = val_df.assign(freq=cal["frequency"].clip(upper=7)).groupby("freq").mean()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(grp.index, grp["actual"], "o-", label="actual")
    ax.plot(grp.index, grp["predicted"], "s--", label="predicted (BG/NBD)")
    ax.set_xlabel("repeat purchases during calibration (7 = 7+)")
    ax.set_ylabel("mean purchases in holdout")
    ax.set_title(f"Holdout check: {validation['holdout_days']} days")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "clv_validation.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
