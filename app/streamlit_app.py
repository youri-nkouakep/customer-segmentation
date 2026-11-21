"""Streamlit dashboard: customer segments, value at stake and recommended actions.

Run locally: streamlit run app/streamlit_app.py
Reads the outputs of scripts/run_pipeline.py (results/*.csv), so no raw data is needed.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
COLORS = {"Champions": "#2a9d8f", "Loyal customers": "#457b9d",
          "Promising newcomers": "#e9c46a", "Hibernating": "#e76f51"}

st.set_page_config(page_title="Customer segmentation", page_icon="📊", layout="wide")


@st.cache_data
def load():
    customers = pd.read_csv(RESULTS / "customers.csv")
    segments = pd.read_csv(RESULTS / "segments.csv", index_col="segment")
    comparison = pd.read_csv(RESULTS / "comparison_k4.csv", index_col="method")
    validation = json.loads((RESULTS / "clv_validation.json").read_text())
    return customers, segments, comparison, validation


customers, segments, comparison, validation = load()
order = list(segments.index)

st.title("Customer segmentation and lifetime value")
st.caption("UCI Online Retail: 541,909 transactions, 4,338 customers, Dec 2010 to Dec 2011. "
           "RFM features, K-means (k = 4), BG/NBD + Gamma-Gamma for 6-month value.")

# ── Key figures ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(customers):,}")
c2.metric("Revenue (12 months)", f"£{customers['Monetary'].sum() / 1e6:.2f}M")
c3.metric("Predicted value, next 6 months", f"£{customers['clv_180d'].sum() / 1e6:.2f}M")
champ = segments.loc["Champions"]
c4.metric("Champions", f"{champ['customer_share']:.0%} of customers",
          f"{champ['revenue_share']:.0%} of revenue", delta_color="off")

tab_seg, tab_cust, tab_method = st.tabs(["Segments and actions", "Customer lookup", "Method"])

# ── Segments ─────────────────────────────────────────────────────────────────
with tab_seg:
    left, right = st.columns([3, 2])
    with left:
        share = segments[["customer_share", "revenue_share"]].mul(100).reset_index().melt(
            id_vars="segment", var_name="metric", value_name="percent")
        share["metric"] = share["metric"].map({"customer_share": "% of customers",
                                               "revenue_share": "% of revenue"})
        fig = px.bar(share, x="segment", y="percent", color="metric", barmode="group",
                     category_orders={"segment": order},
                     color_discrete_sequence=["#bbbbbb", "#2a9d8f"])
        fig.update_layout(height=340, legend_title=None, xaxis_title=None, yaxis_title="%")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.scatter(customers.sample(min(2000, len(customers)), random_state=0),
                         x="Recency", y="Monetary", color="Segment", log_y=True, opacity=0.6,
                         color_discrete_map=COLORS, category_orders={"Segment": order})
        fig.update_layout(height=340, legend_title=None)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Recommended action per segment")
    table = segments.assign(
        customers=segments["customers"].astype(int),
        **{"median recency (days)": segments["recency"],
           "median orders": segments["frequency"],
           "median spend (£)": segments["monetary"].round(0),
           "6-month value (£)": segments["clv_total"].round(0)},
    )[["customers", "median recency (days)", "median orders", "median spend (£)",
       "6-month value (£)", "action"]]
    st.dataframe(table, width="stretch")

# ── Customer lookup ──────────────────────────────────────────────────────────
with tab_cust:
    cid = st.selectbox("Customer ID", customers["CustomerID"].sort_values())
    row = customers.set_index("CustomerID").loc[cid]
    a, b, c, d = st.columns(4)
    a.metric("Segment", row["Segment"])
    b.metric("Last purchase", f"{int(row['Recency'])} days ago")
    c.metric("Orders / spend", f"{int(row['Frequency'])} / £{row['Monetary']:,.0f}")
    d.metric("Predicted value (6 months)", f"£{row['clv_180d']:,.0f}")
    st.info(f"**Action:** {segments.loc[row['Segment'], 'action']}")

# ── Method ───────────────────────────────────────────────────────────────────
with tab_method:
    st.markdown("**Clustering methods compared at k = 4** (log-scaled, standardized RFM).")
    st.dataframe(comparison.drop(columns="k").round(3), width="stretch")
    st.markdown(
        f"**Lifetime value check.** Models fitted until 9 Sep 2011, then compared with the "
        f"next {validation['holdout_days']} days: correlation of {validation['purchases_corr']:.2f} "
        f"between predicted and actual purchases per customer, but the total is under-predicted "
        f"({validation['purchases_predicted_total']:,.0f} vs {validation['purchases_actual_total']:,}) "
        f"because the holdout covers the Christmas peak, which the models do not capture.")
    st.image(str(RESULTS / "figures" / "clv_validation.png"), width=520)
