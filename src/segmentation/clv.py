"""Customer lifetime value: BG/NBD (purchase frequency) + Gamma-Gamma (spend).

Both models are implemented with scipy from the original papers, without the
unmaintained `lifetimes` package:
- Fader, Hardie & Lee (2005), "Counting your customers the easy way"
- Fader & Hardie (2013), "The Gamma-Gamma model of monetary value"

Time unit: days. `frequency` is the number of repeat purchase days, `recency`
the age of the customer at the last purchase, `T` the age at the end of the period.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1


LOG_BOUNDS = (-8.0, 12.0)   # parameters kept in [3e-4, 1.6e5] to avoid numerical overflow


def _fit(nll, starts):
    """Nelder-Mead from several starting points (in log space); keep the best optimum.

    The likelihoods have degenerate local optima (e.g. p -> inf, v -> 0 for the
    Gamma-Gamma model), so a single starting point is not enough.
    """
    def bounded(log_params):
        return nll(np.clip(log_params, *LOG_BOUNDS))

    best = None
    for start in starts:
        res = minimize(bounded, np.log(start), method="Nelder-Mead",
                       options={"maxiter": 40000, "xatol": 1e-10, "fatol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    return np.exp(np.clip(best.x, *LOG_BOUNDS)), best.fun


def summary(transactions, end):
    """Frequency / recency / T / mean repeat spend per customer, up to `end` (exclusive)."""
    tx = transactions[transactions["InvoiceDate"] < end]
    daily = (tx.assign(Day=tx["InvoiceDate"].dt.normalize())
               .groupby(["CustomerID", "Day"])["Revenue"].sum().reset_index())
    g = daily.groupby("CustomerID")
    first, last = g["Day"].min(), g["Day"].max()
    out = pd.DataFrame({
        "frequency": g.size() - 1,
        "recency": (last - first).dt.days,
        "T": (end.normalize() - first).dt.days,
    })
    # Mean spend of repeat purchases only (the first purchase is excluded, as in the paper)
    repeat = daily.merge(first.rename("First"), on="CustomerID")
    repeat = repeat[repeat["Day"] > repeat["First"]]
    out["monetary"] = repeat.groupby("CustomerID")["Revenue"].mean()
    out["monetary"] = out["monetary"].fillna(0.0)
    return out


class BetaGeo:
    """BG/NBD model. Parameters (r, alpha, a, b)."""

    def fit(self, frequency, recency, T):
        x, tx, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))

        def nll(log_params):
            r, alpha, a, b = np.exp(log_params)
            a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
            a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
            a3 = -(r + x) * np.log(alpha + T)
            with np.errstate(divide="ignore"):
                a4 = np.where(x > 0,
                              np.log(a) - np.log(np.maximum(b + x - 1, 1e-12))
                              - (r + x) * np.log(alpha + tx),
                              -np.inf)
            return -np.sum(a1 + a2 + np.logaddexp(a3, a4))

        starts = ([1.0, 10.0, 1.0, 1.0], [0.5, 50.0, 0.5, 2.0], [1.0, 100.0, 0.1, 5.0])
        (self.r, self.alpha, self.a, self.b), self.nll_ = _fit(nll, starts)
        return self

    def _ratio(self, x, tx, T):
        return np.where(x > 0, self.a / (self.b + x - 1) *
                        ((self.alpha + T) / (self.alpha + tx)) ** (self.r + x), 0.0)

    def prob_alive(self, frequency, recency, T):
        x, tx, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        return 1.0 / (1.0 + self._ratio(x, tx, T))

    def expected_purchases(self, t, frequency, recency, T):
        """Expected number of purchases in the next t days."""
        x, tx, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        z = t / (alpha + T + t)
        num = (a + b + x - 1) / (a - 1) * (
            1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp2f1(r + x, b + x, a + b + x - 1, z))
        return num / (1.0 + self._ratio(x, tx, T))


class GammaGamma:
    """Gamma-Gamma spend model. Parameters (p, q, v). Fitted on repeat customers."""

    def fit(self, frequency, monetary):
        x, m = np.asarray(frequency, dtype=float), np.asarray(monetary, dtype=float)
        keep = (x > 0) & (m > 0)
        x, m = x[keep], m[keep]

        def nll(log_params):
            p, q, v = np.exp(log_params)
            ll = (gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(v)
                  + (p * x - 1) * np.log(m) + p * x * np.log(x) - (p * x + q) * np.log(x * m + v))
            return -np.sum(ll)

        mean = float(np.mean(m))
        starts = ([1.0, 1.0, 1.0], [2.0, 5.0, 2.0 * mean], [5.0, 3.0, mean], [10.0, 2.0, 0.1 * mean])
        (self.p, self.q, self.v), self.nll_ = _fit(nll, starts)
        return self

    def expected_spend(self, frequency, monetary):
        """Expected mean spend per purchase (population mean for one-time buyers)."""
        x, m = np.asarray(frequency, dtype=float), np.asarray(monetary, dtype=float)
        population = self.p * self.v / (self.q - 1)
        conditional = self.p * (self.v + x * m) / (self.p * x + self.q - 1)
        return np.where(x > 0, conditional, population)


def holdout_validation(transactions, calibration_end, holdout_end):
    """Fit on the calibration period, compare predicted and actual purchases and revenue
    in the holdout period, for the customers already known at the end of calibration."""
    cal = summary(transactions, calibration_end)
    bg = BetaGeo().fit(cal["frequency"], cal["recency"], cal["T"])
    gg = GammaGamma().fit(cal["frequency"], cal["monetary"])
    days = (holdout_end - calibration_end).days

    hold = transactions[(transactions["InvoiceDate"] >= calibration_end)
                        & (transactions["InvoiceDate"] < holdout_end)]
    hold = hold.assign(Day=hold["InvoiceDate"].dt.normalize())
    actual_n = hold.groupby("CustomerID")["Day"].nunique().reindex(cal.index, fill_value=0)
    actual_rev = hold.groupby("CustomerID")["Revenue"].sum().reindex(cal.index, fill_value=0.0)

    pred_n = bg.expected_purchases(days, cal["frequency"], cal["recency"], cal["T"])
    pred_rev = pred_n * gg.expected_spend(cal["frequency"], cal["monetary"])
    return {
        "customers": len(cal),
        "holdout_days": days,
        "purchases_actual_total": int(actual_n.sum()),
        "purchases_predicted_total": float(pred_n.sum()),
        "purchases_mae": float(np.mean(np.abs(pred_n - actual_n))),
        "purchases_corr": float(np.corrcoef(pred_n, actual_n)[0, 1]),
        "revenue_actual_total": float(actual_rev.sum()),
        "revenue_predicted_total": float(pred_rev.sum()),
        "revenue_corr": float(np.corrcoef(pred_rev, actual_rev)[0, 1]),
    }, pd.DataFrame({"predicted": pred_n, "actual": actual_n})


def predict_clv(transactions, end, horizon_days=180):
    """Fit on all data up to `end` and predict purchases, spend and value for the horizon."""
    s = summary(transactions, end)
    bg = BetaGeo().fit(s["frequency"], s["recency"], s["T"])
    gg = GammaGamma().fit(s["frequency"], s["monetary"])
    s["p_alive"] = bg.prob_alive(s["frequency"], s["recency"], s["T"])
    s["pred_purchases"] = bg.expected_purchases(horizon_days, s["frequency"], s["recency"], s["T"])
    s["pred_spend"] = gg.expected_spend(s["frequency"], s["monetary"])
    s["clv"] = s["pred_purchases"] * s["pred_spend"]
    return s, bg, gg
