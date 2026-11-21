"""Loading and cleaning of the UCI Online Retail dataset."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "online_retail.csv"


def load_raw(path=RAW_PATH):
    """Read the raw transactions (CSV export of the UCI 'Online Retail' dataset)."""
    df = pd.read_csv(path, dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": "Int64"},
                     encoding="utf-8")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed")
    return df


def clean(df):
    """Keep valid, identified sales and add the line revenue.

    Removed: lines without customer ID, cancellations (invoice starting with 'C'),
    non-positive quantities or prices, and exact duplicates.
    Returns the cleaned frame and a dict with the number of rows removed at each step.
    """
    report = {"raw_rows": len(df)}
    df = df.dropna(subset=["CustomerID"])
    report["after_customer_id"] = len(df)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    report["after_cancellations"] = len(df)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    report["after_invalid_qty_price"] = len(df)
    df = df.drop_duplicates()
    report["after_duplicates"] = len(df)
    df = df.assign(Revenue=df["Quantity"] * df["UnitPrice"])
    return df.reset_index(drop=True), report
