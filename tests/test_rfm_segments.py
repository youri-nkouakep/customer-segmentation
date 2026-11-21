import pandas as pd

from segmentation.data import clean
from segmentation.rfm import build_rfm
from segmentation.segments import name_clusters


def transactions():
    return pd.DataFrame({
        "InvoiceNo": ["1", "2", "C3", "4", "5", "5"],
        "StockCode": ["A"] * 6,
        "Description": ["x"] * 6,
        "Quantity": [2, 1, -1, 3, 0, 0],
        "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-10", "2011-01-11",
                                       "2011-01-05", "2011-01-06", "2011-01-06"]),
        "UnitPrice": [5.0, 10.0, 10.0, 1.0, 1.0, 1.0],
        "CustomerID": pd.array([1, 1, 1, 2, 2, None], dtype="Int64"),
        "Country": ["UK"] * 6,
    })


def test_clean_removes_cancellations_and_invalid_rows():
    df, report = clean(transactions())
    assert list(df["InvoiceNo"]) == ["1", "2", "4"]
    assert report["raw_rows"] == 6 and report["after_duplicates"] == 3


def test_build_rfm():
    df, _ = clean(transactions())
    rfm = build_rfm(df, snapshot=pd.Timestamp("2011-01-11"))
    assert rfm.loc[1, "Recency"] == 1 and rfm.loc[1, "Frequency"] == 2
    assert rfm.loc[1, "Monetary"] == 20.0 and rfm.loc[2, "Monetary"] == 3.0


def test_name_clusters():
    prof = pd.DataFrame({"recency": [8, 170, 16, 50], "frequency": [10, 1, 2, 4],
                         "monetary": [3700, 300, 480, 1350]})
    assert name_clusters(prof) == {0: "Champions", 1: "Hibernating",
                                   2: "Promising newcomers", 3: "Loyal customers"}
