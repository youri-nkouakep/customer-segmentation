"""The BG/NBD and Gamma-Gamma implementations must reproduce the published CDNOW results.

Fader, Hardie & Lee (2005): r = 0.243, alpha = 4.414, a = 0.793, b = 2.426, and 1.226
expected purchases in 39 weeks for a customer with x = 2, t_x = 30.43, T = 38.86.
Fader & Hardie (2013): p = 6.25, q = 3.74, gamma = 15.44.
"""
from pathlib import Path

import pandas as pd
import pytest

from segmentation.clv import BetaGeo, GammaGamma

DATA = Path(__file__).parent / "data"


def test_bg_nbd_matches_paper():
    d = pd.read_csv(DATA / "cdnow_summary.csv")
    bg = BetaGeo().fit(d["frequency"], d["recency"], d["T"])
    assert (bg.r, bg.alpha, bg.a, bg.b) == pytest.approx((0.243, 4.414, 0.793, 2.426), abs=2e-3)
    assert float(bg.expected_purchases(39, 2, 30.43, 38.86)) == pytest.approx(1.226, abs=2e-3)


def test_gamma_gamma_matches_paper():
    d = pd.read_csv(DATA / "cdnow_summary_spend.csv")
    d = d[d["x"] > 0]
    gg = GammaGamma().fit(d["x"], d["zbar"])
    assert (gg.p, gg.q, gg.v) == pytest.approx((6.25, 3.74, 15.44), abs=2e-2)
