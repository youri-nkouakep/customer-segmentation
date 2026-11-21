#!/usr/bin/env python3
"""Download the UCI Online Retail dataset to data/raw/online_retail.csv.

Source: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5BW33 (CC BY 4.0)
"""
import io
from pathlib import Path
import urllib.request
import zipfile

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "online_retail.csv"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as r:
        archive = zipfile.ZipFile(io.BytesIO(r.read()))
    with archive.open("Online Retail.xlsx") as f:
        df = pd.read_excel(f, dtype={"InvoiceNo": str, "StockCode": str})
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows to {OUT}")


if __name__ == "__main__":
    main()
