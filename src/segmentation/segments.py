"""Turn clusters into named, actionable customer segments."""
import pandas as pd

ACTIONS = {
    "Champions": "VIP program: early access, referral rewards, no discounts needed",
    "Loyal customers": "Cross-sell and upsell; loyalty points to move them up to Champions",
    "Promising newcomers": "Onboarding sequence and a second-purchase incentive",
    "Hibernating": "Low-cost win-back campaign; remove from expensive channels if no response",
}


def profile(rfm, labels):
    """Median RFM, size and revenue share of each cluster."""
    p = (rfm.assign(cluster=labels)
            .groupby("cluster")
            .agg(customers=("Recency", "size"), recency=("Recency", "median"),
                 frequency=("Frequency", "median"), monetary=("Monetary", "median"),
                 revenue=("Monetary", "sum")))
    p["customer_share"] = p["customers"] / p["customers"].sum()
    p["revenue_share"] = p["revenue"] / p["revenue"].sum()
    return p


def name_clusters(prof):
    """Name 4 clusters from their median profile.

    Value = rank of median frequency + rank of median monetary value.
    Highest value -> Champions, lowest -> Hibernating; of the two middle clusters,
    the more recent one -> Promising newcomers, the other -> Loyal customers.
    """
    if len(prof) != 4:
        return {c: f"Segment {c}" for c in prof.index}
    value = prof["frequency"].rank() + prof["monetary"].rank()
    order = (prof.assign(value=value)
                 .sort_values(["value", "monetary"]).index.tolist())
    names = {order[-1]: "Champions", order[0]: "Hibernating"}
    middle = prof.loc[order[1:3]].sort_values("recency").index.tolist()
    names[middle[0]] = "Promising newcomers"
    names[middle[1]] = "Loyal customers"
    return names


def segment_table(prof, names, clv=None):
    """Summary table: one row per named segment, with its recommended action."""
    t = prof.rename(index=names)
    t.index.name = "segment"
    if clv is not None:
        t = t.join(clv)
    t["action"] = [ACTIONS.get(s, "") for s in t.index]
    order = [s for s in ACTIONS if s in t.index]
    return t.loc[order]
