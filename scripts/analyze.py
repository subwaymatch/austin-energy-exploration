#!/usr/bin/env python3
"""Fraud-detection analysis of the combined Austin Energy ProCard data.

Question: using only the transaction export (no invoices, no interviews), which
standard audit-analytics tests would have put Mark Ybarra's card on a short
review list, and how early?

Reads   data/ae_pcard_transactions.csv, data/ae_pcard_combined.csv
Writes  analysis/results.json           everything the HTML report needs
        analysis/cardholder_scores.csv  one row per cardholder, all test metrics
        analysis/subject_transactions.csv  the subject's transactions with flags

Cardholders other than the subject are shown under aliases (CH-001 = largest
spender) unless --show-names is passed: a high score is a reason to look at
receipts, not a finding about a person.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

SUBJECT = "Ybarra, Mark Anthony"
PERIOD_START, PERIOD_END = "2018-01-01", "2023-09-30"   # newer-system data up to the subject's last purchases
MIN_N = 100                                              # purchases needed before a cardholder is scored
LIMIT = 3000.0                                           # single-purchase limit implied by the data (see find_limit)
RNG = np.random.default_rng(20261114)

BENFORD1 = np.log10(1 + 1 / np.arange(1, 10))
BENFORD2 = np.log10(1 + 1 / np.arange(10, 100))


# ----------------------------------------------------------------------------- helpers
def first_digits(a, k=1):
    """Leading k significant digits of positive amounts (computed on integer cents)."""
    cents = np.round(np.asarray(a, dtype=float) * 100).astype(np.int64)
    s = pd.Series(cents).astype(str).str.lstrip("0")
    if k == 2:
        s = s.where(s.str.len() >= 2, s + "0")
    return s.str[:k].astype(int).to_numpy()


def digit_table(a, k=1):
    d = first_digits(a, k)
    lo, hi = (1, 10) if k == 1 else (10, 100)
    obs = np.bincount(d, minlength=hi)[lo:hi]
    return obs


def mad(obs, expected):
    n = obs.sum()
    return float(np.abs(obs / n - expected).mean()) if n else np.nan


def chi2(obs, expected):
    n = obs.sum()
    stat = float(((obs - n * expected) ** 2 / (n * expected)).sum())
    return stat, float(stats.chi2.sf(stat, len(expected) - 1))


def mad_z(obs, base, sims=400):
    """How unusual is this MAD for a sample of this size drawn from `base`?"""
    n = int(obs.sum())
    draws = RNG.multinomial(n, base, size=sims)
    m = np.abs(draws / n - base).mean(axis=1)
    return float((mad(obs, base) - m.mean()) / m.std())


def vendor_core(v):
    """Rough payee key: drop processor prefixes, punctuation, digits; keep 12 characters."""
    v = re.sub(r"^(PAYPAL|PP|SQU|SQ|TST|IN|INT IN|PY|WPY|SP|DNH|EB|FS)\b[ *]*", "", v.upper())
    v = re.sub(r"^(SQ)\b[ *]*", "", v)
    return re.sub(r"[^A-Z]", "", v)[:12]


def rank_of(series, name, ascending=False):
    r = series.rank(ascending=ascending, method="min")
    return int(r[name]) if name in r.index and not np.isnan(r[name]) else None


# ----------------------------------------------------------------------------- load
def load(data_dir):
    t = pd.read_csv(Path(data_dir) / "ae_pcard_transactions.csv",
                    dtype={"mcc": str, "gl_unit": str, "gl_object": str, "gl_activity": str, "gl_task_order": str,
                           "supplier_city": str, "supplier_state": str})
    for c in ["purchase_date", "post_date", "ch_signoff_date", "mgr_signoff_date"]:
        t[c] = pd.to_datetime(t[c])
    t["vendor_name"] = t["vendor_name"].fillna("")
    # payment facilitator: PayPal / Square (Block) acting between the card and the real payee
    dba = t["dba_name"].fillna("").str.upper()
    city = t["supplier_city"].fillna("").str.lower()
    t["facilitator"] = np.select(
        [t["vendor_name"].str.match(r"^(PAYPAL|PP)\b") | dba.str.startswith(("PAYPAL", "PPINC", "PPI ")) | (city == "4029357733"),
         t["vendor_name"].str.match(r"^(SQ|SQU)\b") | dba.str.startswith(("SQUARE", "BLOCK, INC")) | (city == "gosq.com")],
        ["PayPal", "Square"], default="")
    t["is_fac"] = t["facilitator"] != ""
    t["vendor_key"] = t["vendor_name"].map(vendor_core)
    t["is_purchase"] = t["amount"] > 0
    t["whole_dollar"] = t["is_purchase"] & (np.round(t["amount"] * 100) % 100 == 0)
    t["round_100"] = t["is_purchase"] & (np.round(t["amount"] * 100) % 10000 == 0)
    t["near_limit"] = t["is_purchase"] & (t["amount"] >= 0.9 * LIMIT) & (t["amount"] <= LIMIT)
    t["weekend"] = t["purchase_date"].dt.dayofweek >= 5
    t["quarter"] = t["post_date"].dt.to_period("Q")
    t["month"] = t["post_date"].dt.to_period("M")
    return t


def find_limit(t):
    """Show the cliff in the amount distribution that reveals the single-purchase limit."""
    out = {}
    for label, lo, hi in [("2018-2022", "2018-01-01", "2022-12-31"), ("2024-2026", "2024-01-01", "2026-12-31")]:
        a = t.loc[t.is_purchase & t.post_date.between(lo, hi), "amount"]
        edges = np.arange(1000, 6001, 100)
        out[label] = {"edges": edges.tolist(), "counts": np.histogram(a, edges)[0].tolist(), "n": int(len(a))}
    a = t.loc[t.is_purchase & t.post_date.between("2018-01-01", "2022-12-31"), "amount"]
    out["below_3000"] = int(((a >= 2900) & (a <= 3000)).sum())
    out["above_3000"] = int(((a > 3000) & (a <= 3100)).sum())
    out["share_over_3000_by_year"] = (t[t.is_purchase].groupby(t.post_date.dt.year)["amount"]
                                      .apply(lambda s: float((s > 3000).mean())).round(4).to_dict())
    return out


# ----------------------------------------------------------------------------- cardholder metrics
def cardholder_metrics(t, pool_fd=None, min_n=MIN_N):
    """One row of test metrics per cardholder for the transactions in `t`."""
    p = t[t.is_purchase]
    g = p.groupby("cardholder_name")
    m = pd.DataFrame({
        "n": g.size(),
        "spend": g["amount"].sum(),
        "median_amount": g["amount"].median(),
        "near_limit_n": g["near_limit"].sum(),
        "near_limit_share": g["near_limit"].mean(),
        "near_limit_dollars": p[p.near_limit].groupby("cardholder_name")["amount"].sum(),
        "fac_n": g["is_fac"].sum(),
        "fac_dollars": p[p.is_fac].groupby("cardholder_name")["amount"].sum(),
        "weekend_share": g["weekend"].mean(),
        "weekend_dollars": p[p.weekend].groupby("cardholder_name")["amount"].sum(),
    }).fillna(0.0)
    m["fac_share"] = m["fac_dollars"] / m["spend"]
    big = p[p.amount >= 100]
    gb = big.groupby("cardholder_name")
    m["whole_dollar_share"] = gb["whole_dollar"].mean()
    m["whole_dollar_n"] = gb["whole_dollar"].sum()
    m["round_100_share"] = gb["round_100"].mean()

    # Benford
    if pool_fd is None:
        pool_fd = digit_table(p["amount"]) / len(p)
    fd, f2 = {}, {}
    for name, a in g["amount"]:
        if len(a) >= min_n:
            o1, o2 = digit_table(a), digit_table(a, 2)
            c, pv = chi2(o1, BENFORD1)
            fd[name] = dict(fd_mad=mad(o1, BENFORD1), fd_chi2=c, fd_p=pv, fd_mad_z_benford=mad_z(o1, BENFORD1),
                            fd_mad_z_pool=mad_z(o1, pool_fd), f2_mad=mad(o2, BENFORD2),
                            f2_mad_z=mad_z(o2, BENFORD2), share_first_digit_2=float(o1[1] / o1.sum()))
    m = m.join(pd.DataFrame(fd).T)

    # vendor exclusivity: payees that only this cardholder uses
    vk = p[p.vendor_key != ""].groupby("vendor_key").agg(users=("cardholder_name", "nunique"), n=("amount", "size"),
                                                           dollars=("amount", "sum"))
    sole = vk[(vk.users == 1) & (vk.n >= 3)].index
    ps = p[p.vendor_key.isin(sole)]
    m["sole_vendor_dollars"] = ps.groupby("cardholder_name")["amount"].sum()
    m["sole_vendor_count"] = ps.groupby("cardholder_name")["vendor_key"].nunique()
    m["sole_vendor_share"] = m["sole_vendor_dollars"] / m["spend"]

    # repeated amounts (>= $100): share of purchases made at the cardholder's five most common amounts
    def top5(a):
        return float(a.round(2).value_counts().head(5).sum() / len(a)) if len(a) else np.nan
    m["repeat_amount_share"] = gb["amount"].apply(top5)

    # split purchases: same cardholder + payee within 7 days, each <= limit, together > limit
    sp = p[(p.vendor_key != "") & (p.amount <= LIMIT)].sort_values(["cardholder_name", "vendor_key", "purchase_date"])
    grp = [sp.cardholder_name, sp.vendor_key]
    prev_amt = sp.groupby(grp)["amount"].shift()
    prev_date = sp.groupby(grp)["purchase_date"].shift()
    gap = (sp.purchase_date - prev_date).dt.days
    sp["split_pair"] = (gap <= 7) & (sp.amount + prev_amt > LIMIT)
    m["split_pairs"] = sp.groupby("cardholder_name")["split_pair"].sum()
    m["split_dollars"] = sp[sp.split_pair].groupby("cardholder_name")["amount"].sum()
    # same-day stacking: two or more near-limit purchases on one day (any payee)
    day = p[p.near_limit].groupby(["cardholder_name", "purchase_date"]).size()
    m["stacked_days"] = day[day >= 2].groupby("cardholder_name").size()

    # merchant category vs accounting code: purchases whose MCC rarely goes with that GL object for anyone else
    pair = p.groupby(["mcc", "gl_object"]).size().rename("pair_n").reset_index()
    own = p.groupby(["cardholder_name", "mcc", "gl_object"]).size().rename("own_n").reset_index()
    mcc_n = p.groupby("mcc").size().rename("mcc_n").reset_index()
    own_mcc = p.groupby(["cardholder_name", "mcc"]).size().rename("own_mcc_n").reset_index()
    q = (p[["txn_id", "cardholder_name", "mcc", "gl_object", "amount"]]
         .merge(pair, on=["mcc", "gl_object"], how="left").merge(own, on=["cardholder_name", "mcc", "gl_object"], how="left")
         .merge(mcc_n, on="mcc", how="left").merge(own_mcc, on=["cardholder_name", "mcc"], how="left"))
    others_mcc = q.mcc_n - q.own_mcc_n
    others_pair = q.pair_n - q.own_n
    q["mismatch"] = (others_mcc >= 30) & (others_pair / others_mcc.clip(lower=1) < 0.02) & q.gl_object.notna()
    m["mismatch_dollars"] = q[q.mismatch].groupby("cardholder_name")["amount"].sum()
    m["mismatch_share"] = m["mismatch_dollars"] / m["spend"]

    # statement months near the $25,000 monthly card limit quoted in the auditor's report
    mon = p.groupby(["cardholder_name", "month"])["amount"].sum()
    m["months_over_20k"] = mon[mon >= 20000].groupby("cardholder_name").size()
    m["max_month"] = mon.groupby("cardholder_name").max()

    unit = t.groupby("cardholder_name")["gl_unit_desc"].agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
    m["unit"] = unit
    return m.fillna({c: 0 for c in ["sole_vendor_dollars", "sole_vendor_count", "sole_vendor_share", "split_pairs",
                                    "split_dollars", "stacked_days", "mismatch_dollars", "mismatch_share",
                                    "months_over_20k", "near_limit_dollars", "fac_dollars", "weekend_dollars"]})


TESTS = {  # label, metric, what an auditor needs to run it
    "total_spend": ("Largest total spending (naive baseline)", "spend"),
    "benford_fd": ("Benford first digit (MAD)", "fd_mad"),
    "benford_fd_z": ("Benford first digit, size-adjusted", "fd_mad_z_benford"),
    "benford_f2": ("Benford first two digits (MAD)", "f2_mad"),
    "benford_f2_z": ("Benford first two digits, size-adjusted", "f2_mad_z"),
    "near_limit": ("Just under the purchase limit (count at 90-100% of $3,000)", "near_limit_n"),
    "near_limit_share": ("Just under the purchase limit (share of purchases)", "near_limit_share"),
    "whole_dollar": ("Whole-dollar amounts (share of purchases >= $100)", "whole_dollar_share"),
    "facilitator": ("Paid through PayPal / Square (dollars)", "fac_dollars"),
    "sole_vendor": ("Payees no other cardholder uses (dollars)", "sole_vendor_dollars"),
    "split": ("Split purchases: same payee within 7 days, together over the limit", "split_pairs"),
    "stacked": ("Days with two or more near-limit purchases", "stacked_days"),
    "weekend": ("Weekend purchases (dollars)", "weekend_dollars"),
    "repeat": ("Repeated identical amounts (share at five most common)", "repeat_amount_share"),
    "mismatch": ("Merchant category does not fit the accounting code (dollars)", "mismatch_dollars"),
    "months": ("Months at or above $20,000 (card limit $25,000)", "months_over_20k"),
}
COMPOSITE = ["benford_fd_z", "near_limit", "whole_dollar", "facilitator", "sole_vendor", "split", "weekend", "mismatch"]


def composite(m):
    """Average percentile rank over the pre-specified tests (0-1, higher = more unusual)."""
    e = m[m.n >= MIN_N]
    pct = pd.DataFrame({k: e[TESTS[k][1]].rank(pct=True) for k in COMPOSITE})
    return pct.mean(axis=1), pct


def ml_scores(m):
    e = m[m.n >= MIN_N].copy()
    feats = ["near_limit_share", "whole_dollar_share", "fac_share", "sole_vendor_share", "weekend_share",
             "repeat_amount_share", "mismatch_share", "fd_mad_z_pool"]
    X = e[feats].fillna(0).to_numpy()
    X = np.column_stack([X, np.log10(e["median_amount"].clip(lower=1)), e["split_pairs"] / e["n"]])
    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=500, random_state=7, contamination="auto").fit(Xs)
    e["iforest"] = -iso.score_samples(Xs)
    lof = LocalOutlierFactor(n_neighbors=20).fit(Xs)
    e["lof"] = -lof.negative_outlier_factor_
    return e[["iforest", "lof"]], feats + ["log_median_amount", "split_rate"]


# ----------------------------------------------------------------------------- main analysis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--show-names", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    R = {"subject": SUBJECT, "period": [PERIOD_START, PERIOD_END], "min_n": MIN_N, "limit": LIMIT}

    t = load(a.data)
    per = t[t.post_date.between(PERIOD_START, PERIOD_END)].copy()
    after = t[t.post_date > PERIOD_END]
    y = per[per.cardholder_name == SUBJECT]
    yp = y[y.is_purchase]

    spend_all = t[t.is_purchase].groupby("cardholder_name")["amount"].sum().sort_values(ascending=False)
    alias = {n: f"CH-{i + 1:03d}" for i, n in enumerate(spend_all.index)}
    alias[SUBJECT] = "Ybarra"
    show = (lambda n: n) if a.show_names else (lambda n: alias.get(n, n))

    # ---- overview
    R["overview"] = {
        "works_transactions": int(len(t)), "works_purchases": int(t.is_purchase.sum()),
        "works_credits": int((~t.is_purchase).sum()), "works_net": float(t.amount.sum()),
        "works_cardholders": int(t.cardholder_name.nunique()),
        "period_transactions": int(len(per)), "period_cardholders": int(per.cardholder_name.nunique()),
        "period_spend": float(per[per.is_purchase].amount.sum()),
        "subject_purchases": int(len(yp)), "subject_spend": float(yp.amount.sum()),
        "subject_credits": int((~y.is_purchase).sum()), "subject_net": float(y.amount.sum()),
        "subject_first": str(y.purchase_date.min().date()), "subject_last": str(y.purchase_date.max().date()),
        "subject_vendors": int(yp.vendor_key.nunique()),
    }
    R["limit_evidence"] = find_limit(t)

    # ---- Benford: populations
    pops = {
        "Ybarra": yp.amount,
        "Facility Management peers": per[per.is_purchase & (per.gl_unit_desc == "Facility Management") & (per.cardholder_name != SUBJECT)].amount,
        "All other cardholders": per[per.is_purchase & (per.cardholder_name != SUBJECT)].amount,
        "All cardholders, Oct 2023-Sep 2026": after[after.is_purchase].amount,
    }
    R["benford_pops"] = {}
    for k, v in pops.items():
        o1, o2 = digit_table(v), digit_table(v, 2)
        c1, p1 = chi2(o1, BENFORD1)
        R["benford_pops"][k] = {"n": int(len(v)), "fd": (o1 / o1.sum()).round(5).tolist(), "fd_counts": o1.tolist(),
                                "fd_mad": mad(o1, BENFORD1), "fd_chi2": c1, "fd_p": p1,
                                "f2": (o2 / o2.sum()).round(5).tolist(), "f2_mad": mad(o2, BENFORD2)}
    R["benford_expected"] = {"fd": BENFORD1.round(5).tolist(), "f2": BENFORD2.round(5).tolist()}
    o2 = digit_table(yp.amount, 2)
    n = o2.sum()
    z2 = (o2 / n - BENFORD2) / np.sqrt(BENFORD2 * (1 - BENFORD2) / n)
    top = np.argsort(-z2)[:6]
    R["subject_f2_top"] = [{"digits": int(i + 10), "observed": int(o2[i]), "expected": float(n * BENFORD2[i]), "z": float(z2[i])} for i in top]
    # Benford on the legacy system, as a baseline for how well ProCard amounts fit at all
    comb = pd.read_csv(Path(a.data) / "ae_pcard_combined.csv", usecols=["source_system", "cardholder_name", "fiscal_year",
                       "fiscal_month", "vendor_name", "item_description", "amount", "gl_unit_desc", "gl_object_desc"])
    leg = comb[comb.source_system == "legacy"].copy()
    lo1 = digit_table(leg[leg.amount > 0].amount)
    R["benford_pops"]["Legacy system, FY2013-FY2018 (accounting lines)"] = {
        "n": int(lo1.sum()), "fd": (lo1 / lo1.sum()).round(5).tolist(), "fd_counts": lo1.tolist(), "fd_mad": mad(lo1, BENFORD1),
        "fd_chi2": chi2(lo1, BENFORD1)[0], "fd_p": chi2(lo1, BENFORD1)[1],
        "f2": (digit_table(leg[leg.amount > 0].amount, 2) / lo1.sum()).round(5).tolist(),
        "f2_mad": mad(digit_table(leg[leg.amount > 0].amount, 2), BENFORD2)}

    # ---- cardholder metrics for the whole period
    m = cardholder_metrics(per)
    comp, pct = composite(m)
    m["composite"] = comp
    ml, ml_feats = ml_scores(m)
    m = m.join(ml)
    e = m[m.n >= MIN_N]
    R["eligible"] = int(len(e))
    fm = e[e.unit == "Facility Management"]
    R["eligible_facility"] = int(len(fm))

    tests = []
    for key, (label, col) in TESTS.items():
        s = e[col]
        tests.append({"key": key, "label": label, "metric": col, "rank": rank_of(s, SUBJECT),
                      "rank_facility": rank_of(fm[col], SUBJECT), "value": float(s[SUBJECT]),
                      "median": float(s.median()), "p95": float(s.quantile(0.95)), "max_other": float(s.drop(SUBJECT).max())})
    for key, label in [("composite", "Composite: average percentile over eight tests"),
                       ("iforest", "Isolation forest on ten cardholder features"),
                       ("lof", "Local outlier factor on the same features")]:
        s = e[key]
        tests.append({"key": key, "label": label, "metric": key, "rank": rank_of(s, SUBJECT),
                      "rank_facility": rank_of(fm[key], SUBJECT), "value": float(s[SUBJECT]), "median": float(s.median()),
                      "p95": float(s.quantile(0.95)), "max_other": float(s.drop(SUBJECT).max())})
    R["tests"] = tests
    R["ml_features"] = ml_feats

    # Benford sensitivity to the minimum sample size
    R["benford_min_n"] = []
    for mn in (50, 100, 200, 500):
        mm = cardholder_metrics(per, min_n=mn)
        ee = mm[mm.n >= mn]
        R["benford_min_n"].append({"min_n": mn, "eligible": int(len(ee)), "fd_mad": rank_of(ee.fd_mad, SUBJECT),
                                   "fd_z": rank_of(ee.fd_mad_z_benford, SUBJECT), "f2_mad": rank_of(ee.f2_mad, SUBJECT),
                                   "f2_z": rank_of(ee.f2_mad_z, SUBJECT)})

    # top of each ranking, for charts (aliased)
    def top_table(col, k=12, extra=()):
        s = e.sort_values(col, ascending=False).head(k)
        return [{"who": show(i), "value": float(r[col]), "n": int(r["n"]), "unit": r["unit"],
                 **{x: float(r[x]) for x in extra}} for i, r in s.iterrows()]
    R["top"] = {k: top_table(TESTS[k][1]) for k in TESTS}
    pur = per[per.is_purchase]
    for row, name in zip(R["top"]["benford_fd"], e.sort_values("fd_mad", ascending=False).head(12).index):
        a_ = pur[pur.cardholder_name == name]
        row["top_vendor_share"] = float(a_.vendor_key.value_counts(normalize=True).iloc[0])
        row["top_amount_share"] = float(a_.amount.round(2).value_counts(normalize=True).iloc[0])
        row["fac_share"] = float(e.loc[name, "fac_share"])
    R["top"]["composite"] = top_table("composite", 15)
    R["top"]["iforest"] = top_table("iforest", 15)
    R["scatter"] = [{"who": show(i), "n": int(r.n), "fd_mad": float(r.fd_mad), "fd_z": float(r.fd_mad_z_benford),
                     "near": float(r.near_limit_share), "fac": float(r.fac_share), "spend": float(r.spend),
                     "whole": float(r.whole_dollar_share), "sole": float(r.sole_vendor_share), "composite": float(r.composite),
                     "facility": r.unit == "Facility Management"} for i, r in e.iterrows()]
    R["composite_pct_subject"] = {k: float(pct.loc[SUBJECT, k]) for k in COMPOSITE}

    # ---- when would each test have flagged him?  (cumulative data up to each quarter end)
    timeline = []
    for qend in pd.period_range("2018Q2", "2023Q3", freq="Q"):
        cut = per[per.post_date <= qend.end_time]
        mm = cardholder_metrics(cut)
        ee = mm[mm.n >= MIN_N]
        row = {"quarter": str(qend), "eligible": int(len(ee)), "subject_n": int(mm.loc[SUBJECT, "n"]) if SUBJECT in mm.index else 0}
        if SUBJECT in ee.index:
            cc, _ = composite(mm)
            ee = ee.assign(composite=cc)
            for key in ["total_spend", "benford_fd", "benford_fd_z", "benford_f2_z", "near_limit", "whole_dollar",
                        "facilitator", "sole_vendor", "split", "weekend", "mismatch"]:
                row[key] = rank_of(ee[TESTS[key][1]], SUBJECT)
            row["composite"] = rank_of(ee["composite"], SUBJECT)
        timeline.append(row)
    R["timeline"] = timeline
    # trailing 12 months, eligible at >= 50 purchases: what an annual review would have seen
    annual = []
    for fy in range(2018, 2024):
        cut = per[per.fiscal_year == fy]
        mm = cardholder_metrics(cut, min_n=50)
        ee = mm[mm.n >= 50]
        if SUBJECT in ee.index:
            annual.append({"fy": fy, "eligible": int(len(ee)), "subject_n": int(ee.loc[SUBJECT, "n"]),
                           **{k: rank_of(ee[TESTS[k][1]], SUBJECT) for k in ["total_spend", "benford_fd", "benford_fd_z",
                              "near_limit", "whole_dollar", "facilitator", "sole_vendor", "split", "weekend", "mismatch"]}})
    R["annual"] = annual

    # ---- the subject up close
    R["subject_by_year"] = [{"year": int(k), "n": int(len(v)), "spend": float(v.amount.sum()),
                             "fac_dollars": float(v[v.is_fac].amount.sum()), "fac_n": int(v.is_fac.sum()),
                             "near_n": int(v.near_limit.sum())} for k, v in yp.groupby(yp.post_date.dt.year)]
    R["subject_fac_by_fy"] = [{"fy": int(k), "dollars": float(v.amount.sum()), "n": int(len(v))}
                              for k, v in yp[yp.is_fac].groupby("fiscal_year")]
    edges = np.arange(0, 3101, 100)
    R["hist"] = {"edges": edges.tolist(),
                 "subject": np.histogram(yp.amount, edges)[0].tolist(),
                 "facility": np.histogram(pops["Facility Management peers"], edges)[0].tolist(),
                 "others": np.histogram(pops["All other cardholders"], edges)[0].tolist()}
    R["subject_split"] = {
        "fac_n": int(yp.is_fac.sum()), "fac_dollars": float(yp[yp.is_fac].amount.sum()),
        "fac_payees": int(yp[yp.is_fac].vendor_key.nunique()),
        "fac_whole_dollar_share": float(yp[yp.is_fac].whole_dollar.mean()),
        "fac_near_limit_share": float(yp[yp.is_fac].near_limit.mean()),
        "fac_median": float(yp[yp.is_fac].amount.median()),
        "fac_weekend_share": float(yp[yp.is_fac].weekend.mean()),
        "other_n": int((~yp.is_fac).sum()), "other_dollars": float(yp[~yp.is_fac].amount.sum()),
        "other_whole_dollar_share": float(yp[~yp.is_fac & (yp.amount >= 100)].whole_dollar.mean()),
        "other_near_limit_share": float(yp[~yp.is_fac].near_limit.mean()),
        "other_median": float(yp[~yp.is_fac].amount.median()),
        "other_weekend_share": float(yp[~yp.is_fac].weekend.mean()),
        "exact_2900": int((yp.amount == 2900).sum()), "top_amounts": yp.amount.round(2).value_counts().head(8).to_dict(),
        "near_n": int(yp.near_limit.sum()), "near_dollars": float(yp[yp.near_limit].amount.sum()),
        "weekend_n": int(yp.weekend.sum()), "weekend_dollars": float(yp[yp.weekend].amount.sum()),
        "over_limit_n": int((yp.amount > LIMIT).sum()),
    }
    by_mgr = y[y.is_purchase & y.is_fac].groupby(y.anon_manager.fillna("(no sign-off)"))
    R["subject_fac_by_manager"] = [{"manager": k, "n": int(len(v)), "dollars": float(v.amount.sum()),
                                    "first": str(v.post_date.min().date()), "last": str(v.post_date.max().date())} for k, v in by_mgr]
    # payees: how many other cardholders ever paid each of his PayPal / Square payees?
    fac_all = t[t.is_purchase & t.is_fac]
    users = fac_all.groupby("vendor_key")["cardholder_name"].nunique()
    yv = yp[yp.is_fac].groupby("vendor_key").agg(n=("amount", "size"), dollars=("amount", "sum"), first=("purchase_date", "min"),
                                                 last=("purchase_date", "max"), descriptor=("vendor_name", "first"),
                                                 mcc=("mcc_description", "first"), median=("amount", "median"))
    yv["other_users"] = users.reindex(yv.index).fillna(1).astype(int) - 1
    yv = yv.sort_values("dollars", ascending=False)
    proc = yp[yp.is_fac].groupby("vendor_key")["facilitator"].first()
    if not a.show_names:  # payees are third parties the auditor's report does not name
        yv["descriptor"] = [f"Payee {i + 1:02d} ({proc[k]})" for i, k in enumerate(yv.index)]
    R["subject_payees"] = [{"descriptor": r.descriptor, "n": int(r.n), "dollars": float(r.dollars), "median": float(r["median"]),
                            "first": str(r["first"].date()), "last": str(r["last"].date()), "mcc": r.mcc,
                            "other_users": int(r.other_users)} for _, r in yv.iterrows()]
    R["subject_payee_summary"] = {"payees": int(len(yv)), "exclusive": int((yv.other_users == 0).sum()),
                                  "exclusive_dollars": float(yv[yv.other_users == 0].dollars.sum())}
    shared = fac_all[fac_all.vendor_key.isin(yv.index) & (fac_all.cardholder_name != SUBJECT)]
    R["shared_payees"] = [{"who": show(k), "n": int(len(v)), "dollars": float(v.amount.sum()), "payees": int(v.vendor_key.nunique())}
                          for k, v in sorted(shared.groupby("cardholder_name"), key=lambda kv: -kv[1].amount.sum())]
    # the same payees on other cards: who else paid them, in which unit, and did it continue after he left?
    oth = fac_all[fac_all.vendor_key.isin(yv.index) & (fac_all.cardholder_name != SUBJECT)]
    R["shared_summary"] = {
        "payees_shared": int((yv.other_users > 0).sum()), "payees_total": int(len(yv)),
        "other_cardholders": int(oth[oth.post_date <= PERIOD_END].cardholder_name.nunique()),
        "other_dollars": float(oth[oth.post_date <= PERIOD_END].amount.sum()),
        "other_n": int((oth.post_date <= PERIOD_END).sum()),
        "other_units": oth[oth.post_date <= PERIOD_END].groupby("gl_unit_desc")["amount"].sum().round(2).sort_values(ascending=False).to_dict(),
        "other_near_limit_share": float(oth[oth.post_date <= PERIOD_END].near_limit.mean()),
        "other_whole_dollar_share": float(oth[oth.post_date <= PERIOD_END].whole_dollar.mean()),
        "after_n": int((fac_all.vendor_key.isin(yv.index) & (fac_all.post_date > PERIOD_END)).sum()),
        "after_dollars": float(fac_all[fac_all.vendor_key.isin(yv.index) & (fac_all.post_date > PERIOD_END)].amount.sum()),
        "by_year": [{"year": int(k), "subject": float(v[v.cardholder_name == SUBJECT].amount.sum()),
                     "others": float(v[v.cardholder_name != SUBJECT].amount.sum())}
                    for k, v in fac_all[fac_all.vendor_key.isin(yv.index)].groupby(fac_all.post_date.dt.year)],
    }
    # PayPal / Square spending by work unit over the period
    home = per.groupby("cardholder_name")["gl_unit_desc"].agg(lambda s: s.mode().iat[0] if s.notna().any() else "(none)")
    pu = per[per.is_purchase].assign(home=per.cardholder_name.map(home))
    u = pu.groupby("home").agg(spend=("amount", "sum"), cardholders=("cardholder_name", "nunique"))
    u["fac"] = pu[pu.is_fac].groupby("home")["amount"].sum()
    u = u.fillna(0).sort_values("fac", ascending=False)
    R["fac_by_unit"] = [{"unit": k, "fac": float(r.fac), "spend": float(r.spend), "cardholders": int(r.cardholders)}
                        for k, r in u.head(8).iterrows()]
    R["fac_total_period"] = float(u.fac.sum())
    # merchant category vs work described
    R["subject_mcc"] = [{"mcc": k, "n": int(len(v)), "dollars": float(v.amount.sum())}
                        for k, v in sorted(yp[yp.is_fac].groupby("mcc_description"), key=lambda kv: -kv[1].amount.sum())][:8]
    R["subject_object"] = [{"object": k, "n": int(len(v)), "dollars": float(v.amount.sum())}
                           for k, v in sorted(yp[yp.is_fac].groupby("gl_object_desc"), key=lambda kv: -kv[1].amount.sum())][:5]
    # monthly spend against the monthly limit
    mon = yp.groupby("month").agg(total=("amount", "sum"), fac=("amount", lambda s: s[yp.loc[s.index, "is_fac"]].sum()))
    mon = mon.reindex(pd.period_range("2018-01", "2023-09", freq="M"), fill_value=0)
    R["subject_monthly"] = [{"month": str(k), "total": float(r.total), "fac": float(r.fac)} for k, r in mon.iterrows()]
    days = yp[yp.near_limit].groupby("purchase_date").agg(n=("amount", "size"), dollars=("amount", "sum"), payees=("vendor_key", "nunique"))
    R["subject_stacked"] = {"days": int((days.n >= 2).sum()), "max_n": int(days.n.max()), "max_dollars": float(days.dollars.max()),
                            "max_day": str(days.dollars.idxmax().date())}

    # ---- legacy system: was the pattern visible in his first months on the card?
    lg = leg[leg.amount > 0].copy()
    lg["is_fac"] = lg.vendor_name.fillna("").str.match(r"^(PAYPAL|PP|SQ|SQU)\b")
    lg["period"] = lg.fiscal_year.astype(int) * 100 + lg.fiscal_month.astype(int)
    ly = lg[lg.cardholder_name == SUBJECT]
    win = lg[lg.period >= ly.period.min()]
    by = win.groupby("cardholder_name").agg(n=("amount", "size"), spend=("amount", "sum"),
                                             fac=("amount", lambda s: s[win.loc[s.index, "is_fac"]].sum()))
    R["legacy"] = {"subject_lines": int(len(ly)), "subject_spend": float(ly.amount.sum()),
                   "subject_fac": float(ly[ly.is_fac].amount.sum()), "subject_fac_n": int(ly.is_fac.sum()),
                   "periods": sorted(int(x) for x in ly.period.unique()), "cardholders": int(len(by)),
                   "fac_rank": rank_of(by.fac, SUBJECT), "spend_rank": rank_of(by.spend, SUBJECT),
                   "fac_total_all": float(by.fac.sum()),
                   "fac_share_by_fy": (lg.groupby("fiscal_year").apply(lambda d: float(d[d.is_fac].amount.sum() / d.amount.sum()), include_groups=False)
                                       .round(4).to_dict())}

    # ---- facilitator use over time, whole department
    q = t[t.is_purchase].assign(year=t.post_date.dt.year)
    R["fac_by_year"] = [{"year": int(k), "subject": float(v[(v.cardholder_name == SUBJECT) & v.is_fac].amount.sum()),
                         "others": float(v[(v.cardholder_name != SUBJECT) & v.is_fac].amount.sum()),
                         "total_spend": float(v.amount.sum())} for k, v in q.groupby("year")]

    # ---- approvals
    s = per[per.mgr_signoff_date.notna()].copy()
    s["lag"] = (s.mgr_signoff_date - s.post_date).dt.days
    s = s.sort_values(["anon_manager", "mgr_signoff_date"])
    s["gap"] = s.groupby("anon_manager")["mgr_signoff_date"].diff().dt.total_seconds()
    mg = s.groupby("anon_manager").agg(n=("amount", "size"), cardholders=("cardholder_name", "nunique"),
                                       dollars=("amount", "sum"), median_lag=("lag", "median"),
                                       within_10s=("gap", lambda g: float((g <= 10).mean())),
                                       median_gap=("gap", "median"))
    mg = mg[mg.n >= 200]
    ym = y[y.mgr_signoff_date.notna()].groupby("anon_manager").size().sort_values(ascending=False)
    R["approvals"] = {"managers": int(len(mg)),
                      "median_within_10s": float(mg.within_10s.median()),
                      "subject_managers": [{"manager": k, "n_subject": int(v), "n_all": int(mg.loc[k, "n"]) if k in mg.index else None,
                                            "cardholders": int(mg.loc[k, "cardholders"]) if k in mg.index else None,
                                            "within_10s": float(mg.loc[k, "within_10s"]) if k in mg.index else None,
                                            "median_gap": float(mg.loc[k, "median_gap"]) if k in mg.index else None,
                                            "median_lag": float(mg.loc[k, "median_lag"]) if k in mg.index else None,
                                            "rank_within_10s": rank_of(mg.within_10s, k) if k in mg.index else None}
                                           for k, v in ym.items()],
                      "subject_no_signoff": int(y.mgr_signoff_date.isna().sum())}

    # ---- transaction-level review queue: count of simple red flags per purchase
    pp = per[per.is_purchase].copy()
    vk = pp[pp.vendor_key != ""].groupby("vendor_key")["cardholder_name"].nunique()
    pp["sole_payee"] = pp.vendor_key.map(vk).fillna(1).eq(1)
    flags = ["near_limit", "whole_dollar", "is_fac", "sole_payee", "weekend"]
    pp["n_flags"] = pp[flags].sum(axis=1)
    queue = []
    for k in range(0, 6):
        sel = pp[pp.n_flags >= k]
        queue.append({"min_flags": k, "n": int(len(sel)), "dollars": float(sel.amount.sum()),
                      "cardholders": int(sel.cardholder_name.nunique()),
                      "subject_n": int((sel.cardholder_name == SUBJECT).sum()),
                      "subject_dollars": float(sel[sel.cardholder_name == SUBJECT].amount.sum()),
                      "subject_fac_dollars": float(sel[(sel.cardholder_name == SUBJECT) & sel.is_fac].amount.sum())})
    R["queue"] = queue
    R["queue_flags"] = ["Amount at 90-100% of the $3,000 limit", "Whole-dollar amount", "Paid through PayPal or Square",
                        "Payee used by no other cardholder", "Purchased on a weekend"]
    sel = pp[pp.n_flags >= 4]
    R["queue4_by_cardholder"] = [{"who": show(k), "n": int(v)} for k, v in sel.cardholder_name.value_counts().head(8).items()]

    # ---- outputs
    cols = ["n", "spend", "unit", "fd_mad", "fd_mad_z_benford", "fd_mad_z_pool", "f2_mad", "f2_mad_z", "share_first_digit_2",
            "near_limit_n", "near_limit_share", "whole_dollar_share", "fac_dollars", "fac_share", "sole_vendor_dollars",
            "sole_vendor_share", "split_pairs", "stacked_days", "weekend_dollars", "weekend_share", "repeat_amount_share",
            "mismatch_dollars", "mismatch_share", "months_over_20k", "max_month", "composite", "iforest", "lof"]
    sc = m.loc[m.n >= MIN_N, cols].sort_values("composite", ascending=False)
    sc.insert(0, "cardholder", [show(i) for i in sc.index])
    sc.round(5).to_csv(out / "cardholder_scores.csv", index=False)
    ys = pp[pp.cardholder_name == SUBJECT][["txn_id", "purchase_date", "post_date", "vendor_name", "amount", "all_item_descriptions",
                                            "gl_object_desc", "mcc_description", "anon_manager", "facilitator"] + flags + ["n_flags"]]
    ys.sort_values("purchase_date").to_csv(out / "subject_transactions.csv", index=False)
    (out / "results.json").write_text(json.dumps(R, indent=1, default=str))

    # ---- console summary
    print(f"eligible cardholders (>= {MIN_N} purchases, {PERIOD_START}..{PERIOD_END}): {len(e)}; facility management: {len(fm)}")
    print(f"subject: {len(yp)} purchases, ${yp.amount.sum():,.2f}; PayPal/Square ${yp[yp.is_fac].amount.sum():,.2f} ({yp.is_fac.sum()} purchases)")
    for r in tests:
        print(f"  {r['label'][:68]:68s} rank {r['rank']:>3} / {len(e)}   (facility {r['rank_facility']}/{len(fm)})  value {r['value']:,.4g}  median {r['median']:,.4g}")


if __name__ == "__main__":
    main()
