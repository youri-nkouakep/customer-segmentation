"""RFM features: Recency, Frequency, Monetary value."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]


def build_rfm(transactions, snapshot=None):
    """One row per customer.

    Recency   : days since the last purchase (at the snapshot date)
    Frequency : number of distinct invoices
    Monetary  : total revenue
    """
    if snapshot is None:
        snapshot = transactions["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = transactions.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        Country=("Country", lambda s: s.mode().iat[0]),
    )
    rfm["Recency"] = (snapshot - rfm["LastPurchase"]).dt.days
    return rfm[["Recency", "Frequency", "Monetary", "Country"]]


def scale_rfm(rfm):
    """Log-transform (the three variables are heavily right-skewed) then standardize."""
    X = np.log1p(rfm[RFM_COLUMNS].to_numpy(dtype=float))
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler
