#!/usr/bin/env python3
"""Render the self-contained HTML report from analysis/results.json.

    python scripts/build_report.py  [--analysis analysis] [--data data] [--out reports/...html]
"""
import argparse
import html
import json
from pathlib import Path

import pandas as pd

W = 760  # chart width in SVG units


def esc(s):
    return html.escape(str(s), quote=True)


def usd(x, compact=False):
    if compact:
        if abs(x) >= 1e6:
            return f"${x / 1e6:.2f}M"
        if abs(x) >= 1e4:
            return f"${x / 1e3:.0f}K"
    return f"${x:,.0f}"


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


# ----------------------------------------------------------------------------- SVG helpers
class Svg:
    def __init__(self, w, h, label):
        self.w, self.h = w, h
        self.p = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">']

    def text(self, x, y, t, cls="tick", anchor="start"):
        self.p.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{esc(t)}</text>')

    def line(self, x1, y1, x2, y2, cls="grid"):
        self.p.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"/>')

    def col(self, x, y, w, h, cls, tip, r=4):
        """Column growing up from the baseline, rounded at the data end only."""
        if h <= 0:
            return
        r = min(r, w / 2, h)
        d = f"M{x:.1f},{y + h:.1f}V{y + r:.1f}Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f}H{x + w - r:.1f}Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y + h:.1f}Z"
        self.p.append(f'<path d="{d}" class="mark {cls}" tabindex="0" data-tip="{esc(tip)}"/>')

    def rect(self, x, y, w, h, cls, tip):
        if h > 0 and w > 0:
            self.p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" class="mark {cls}" tabindex="0" data-tip="{esc(tip)}"/>')

    def hbar(self, x, y, w, h, cls, tip, r=4):
        if w <= 0:
            return
        r = min(r, h / 2, w)
        d = f"M{x:.1f},{y:.1f}H{x + w - r:.1f}Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y + h - r:.1f}Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f}H{x:.1f}Z"
        self.p.append(f'<path d="{d}" class="mark {cls}" tabindex="0" data-tip="{esc(tip)}"/>')

    def dot(self, x, y, r, cls, tip):
        self.p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="mark dot {cls}" tabindex="0" data-tip="{esc(tip)}"/>')

    def path(self, pts, cls):
        d = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.p.append(f'<path d="{d}" class="{cls}"/>')

    def yaxis(self, x0, x1, y0, y1, vmax, ticks, fmt):
        for v in ticks:
            y = y1 - (y1 - y0) * v / vmax
            self.line(x0, y, x1, y, "axis" if v == 0 else "grid")
            self.text(x0 - 8, y + 4, fmt(v), "tick", "end")

    def done(self):
        return "".join(self.p) + "</svg>"


def legend(items):
    out = []
    for kind, cls, label in items:
        out.append(f'<span class="key"><span class="sw {kind} {cls}"></span>{esc(label)}</span>')
    return '<div class="legend">' + "".join(out) + "</div>"


def table(headers, rows, cls="", align=None):
    align = align or ["l"] + ["r"] * (len(headers) - 1)
    th = "".join(f'<th class="{a}">{h}</th>' for h, a in zip(headers, align))
    body = "".join("<tr>" + "".join(f'<td class="{a}">{c}</td>' for c, a in zip(r, align)) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(num, title, sub, body, caption, data=None, leg=""):
    d = f"<details><summary>Data table</summary>{data}</details>" if data else ""
    return (f'<figure class="card" id="fig{num}"><h3>Figure {num}. {title}</h3><p class="sub">{sub}</p>{leg}{body}'
            f'<figcaption>{caption}</figcaption>{d}</figure>')


# ----------------------------------------------------------------------------- charts
def fig_first_digit(R):
    exp = R["benford_expected"]["fd"]
    yb = R["benford_pops"]["Ybarra"]["fd"]
    fm = R["benford_pops"]["Facility Management peers"]["fd"]
    ot = R["benford_pops"]["All other cardholders"]["fd"]
    h, x0, x1, y0, y1 = 320, 48, W - 12, 24, 280
    s = Svg(W, h, "First-digit distribution: Ybarra, Facility Management peers and Benford's Law")
    vmax = 0.55
    s.yaxis(x0, x1, y0, y1, vmax, [0, .1, .2, .3, .4, .5], lambda v: pct(v))
    band = (x1 - x0) / 9
    for i in range(9):
        cx = x0 + band * (i + .5)
        for j, (vals, cls, name) in enumerate([(yb, "s1", "Ybarra"), (fm, "s2", "Facility Management peers")]):
            hh = (y1 - y0) * vals[i] / vmax
            s.col(cx - 21 + j * 22, y1 - hh, 20, hh, cls, f"First digit {i + 1} · {name}: {pct(vals[i], 1)} · Benford: {pct(exp[i], 1)}")
        ye = y1 - (y1 - y0) * exp[i] / vmax
        s.line(cx - 27, ye, cx + 27, ye, "ref")
        s.text(cx, y1 + 18, str(i + 1), "tick", "middle")
    s.text(x0 + band * 1.5 - 11, y1 - (y1 - y0) * yb[1] / vmax - 7, pct(yb[1], 1), "val", "middle")
    s.text((x0 + x1) / 2, h - 4, "First digit of the purchase amount", "axl", "middle")
    rows = [[i + 1, pct(exp[i], 1), pct(yb[i], 1), pct(fm[i], 1), pct(ot[i], 1)] for i in range(9)]
    return figure(1, "Half of Ybarra's purchases start with a 2",
                  f"Share of purchases by first digit, January 2018–September 2023. Ybarra: {R['benford_pops']['Ybarra']['n']:,} purchases; "
                  f"Facility Management peers: {R['benford_pops']['Facility Management peers']['n']:,}.",
                  s.done(),
                  "Benford's Law expects 17.6% of amounts to start with 2. The peers in his own work unit also run high on 2 (26.5%), "
                  "which is the first hint that the unit, and not only one card, buys a lot at $2,000–$2,999.",
                  table(["First digit", "Benford", "Ybarra", "Facility peers", "All other cardholders"], rows),
                  legend([("box", "s1", "Ybarra"), ("box", "s2", "Facility Management peers"), ("ln", "refk", "Benford's Law")]))


def fig_two_digit(R):
    exp = R["benford_expected"]["f2"]
    yb = R["benford_pops"]["Ybarra"]["f2"]
    h, x0, x1, y0, y1 = 300, 48, W - 12, 24, 260
    vmax = 0.17
    s = Svg(W, h, "First-two-digit distribution of Ybarra's purchases against Benford's Law")
    s.yaxis(x0, x1, y0, y1, vmax, [0, .05, .10, .15], lambda v: pct(v))
    band = (x1 - x0) / 90
    for i in range(90):
        hh = (y1 - y0) * yb[i] / vmax
        s.col(x0 + band * i + 1, y1 - hh, band - 2, hh, "s1", f"Amounts starting {i + 10} · Ybarra: {pct(yb[i], 1)} · Benford: {pct(exp[i], 1)}", r=2)
        if (i + 10) % 10 == 0:
            s.text(x0 + band * (i + .5), y1 + 18, str(i + 10), "tick", "middle")
    s.path([(x0 + band * (i + .5), y1 - (y1 - y0) * exp[i] / vmax) for i in range(90)], "refline")
    for d in (28, 29):
        i = d - 10
        s.text(x0 + band * (i + .5) + (6 if d == 29 else -6), y1 - (y1 - y0) * yb[i] / vmax - 6, f"{d}: {pct(yb[i], 1)}", "val", "start" if d == 29 else "end")
    s.text((x0 + x1) / 2, h - 4, "First two digits of the purchase amount", "axl", "middle")
    top = R["subject_f2_top"]
    rows = [[t["digits"], t["observed"], f"{t['expected']:.1f}", f"{t['z']:.1f}"] for t in top]
    return figure(2, "The excess sits almost entirely on 28 and 29",
                  "Share of Ybarra's 973 purchases by first two digits.",
                  s.done(),
                  "This is where Benford earns its place: not as a verdict, but as a pointer. Amounts beginning 28 or 29 are "
                  "almost all $2,800–$2,999, which leads straight to the purchase limit.",
                  table(["First two digits", "Observed", "Expected", "z-score"], rows),
                  legend([("box", "s1", "Ybarra"), ("ln", "refk", "Benford's Law")]))


def fig_limit(R):
    ev = R["limit_evidence"]
    panels = [("2018-2022", "Posted 2018–2022"), ("2024-2026", "Posted 2024–2026")]
    h = 250
    s = Svg(W, h, "Number of purchases by amount, showing a cliff at $3,000 and later at $5,000")
    pw = (W - 30) / 2
    for k, (key, title) in enumerate(panels):
        d = ev[key]
        ox = k * (pw + 30)
        x0, x1, y0, y1 = ox + 46, ox + pw, 30, 205
        vmax = max(d["counts"]) * 1.08
        step = 500 if vmax > 1200 else 200
        ticks = list(range(0, int(vmax) + 1, step))
        s.yaxis(x0, x1, y0, y1, vmax, ticks, lambda v: f"{v:,.0f}")
        band = (x1 - x0) / len(d["counts"])
        for i, c in enumerate(d["counts"]):
            hh = (y1 - y0) * c / vmax
            lo = d["edges"][i]
            s.col(x0 + band * i + 1, y1 - hh, band - 2, hh, "s1", f"${lo:,}–${lo + 99:,}: {c:,} purchases", r=2)
        for v in (1000, 2000, 3000, 4000, 5000, 6000):
            s.text(x0 + band * (v - 1000) / 100, y1 + 16, f"${v // 1000}K", "tick", "middle")
        s.text(x0, 16, title, "lbl", "start")
    return figure(3, "The data shows where the single-purchase limit was",
                  "Number of purchases by amount, $100 bins from $1,000 to $6,000, all Austin Energy cardholders.",
                  s.done(),
                  f"From 2018 to 2022, {ev['below_3000']:,} purchases fall at $2,900–$3,000 and only {ev['above_3000']} at $3,000.01–$3,100. "
                  "From 2023 purchases above $3,000 become common and a second cliff appears at $5,000, which suggests the limit was raised for some cards. "
                  "No limit is stated anywhere in the release, so this is an inference from the amounts; the City should confirm it and when it changed.",
                  table(["Bin starts at", "2018–2022", "2024–2026"],
                        [[f"${e:,}", f"{a:,}", f"{b:,}"] for e, a, b in zip(ev["2018-2022"]["edges"], ev["2018-2022"]["counts"], ev["2024-2026"]["counts"])]))


def fig_hist(R):
    hst = R["hist"]
    ny, nf = R["overview"]["subject_purchases"], R["benford_pops"]["Facility Management peers"]["n"]
    yb = [c / ny for c in hst["subject"]]
    fm = [c / nf for c in hst["facility"]]
    h, x0, x1, y0, y1 = 310, 48, W - 12, 24, 265
    vmax = 0.30
    s = Svg(W, h, "Purchase amounts for Ybarra and Facility Management peers")
    s.yaxis(x0, x1, y0, y1, vmax, [0, .1, .2, .3], lambda v: pct(v))
    band = (x1 - x0) / len(yb)
    for i, v in enumerate(yb):
        hh = (y1 - y0) * v / vmax
        lo = hst["edges"][i]
        s.col(x0 + band * i + 2, y1 - hh, band - 4, hh, "s1",
              f"${lo:,}–${lo + 99:,} · Ybarra: {hst['subject'][i]} purchases ({pct(v, 1)}) · Facility peers: {pct(fm[i], 1)}", r=3)
    pts = []
    for i, v in enumerate(fm):
        y = y1 - (y1 - y0) * v / vmax
        pts += [(x0 + band * i, y), (x0 + band * (i + 1), y)]
    s.path(pts, "s2line")
    for v in range(0, 3001, 500):
        s.text(x0 + band * v / 100, y1 + 18, f"${v:,}", "tick", "middle")
    i28 = 28
    s.text(x0 + band * (i28 + 1), y1 - (y1 - y0) * max(yb[28], yb[29]) / vmax - 8,
           f"{hst['subject'][28] + hst['subject'][29]} purchases at $2,800–$2,999", "val", "end")
    s.text((x0 + x1) / 2, h - 4, "Purchase amount ($100 bins)", "axl", "middle")
    rows = [[f"${e:,}–${e + 99:,}", a, pct(a / ny, 1), pct(b / nf, 1)] for e, a, b in zip(hst["edges"], hst["subject"], hst["facility"])]
    return figure(4, "Two different cards in one: small routine purchases, and a block just under $3,000",
                  "Share of each group's purchases by amount, January 2018–September 2023. Peer purchases above $3,100 stay in the denominator.",
                  s.done(),
                  f"{R['subject_split']['near_n']} of his purchases ({pct(R['subject_split']['near_n'] / ny)}) are at $2,700–$3,000, worth "
                  f"{usd(R['subject_split']['near_dollars'])}. None is above $3,000. {R['subject_split']['exact_2900']} are exactly $2,900.",
                  table(["Amount", "Ybarra purchases", "Ybarra share", "Facility peers share"], rows),
                  legend([("box", "s1", "Ybarra"), ("ln", "s2k", "Facility Management peers")]))


def fig_fac_rank(R):
    top = R["top"]["facilitator"][:12]
    rowh, x0, x1 = 26, 150, W - 90
    h = 20 + rowh * len(top)
    vmax = top[0]["value"] * 1.02
    s = Svg(W, h, "Dollars paid through PayPal or Square, top twelve cardholders")
    for i, t in enumerate(top):
        y = 10 + rowh * i
        me = t["who"] == "Ybarra"
        name = t["who"] + ("" if me or t["unit"] != "Facility Management" else " (FM)")
        s.text(x0 - 10, y + 15, name, "val" if me else "lbl", "end")
        w = (x1 - x0) * t["value"] / vmax
        s.hbar(x0, y + 3, w, 16, "s1" if me else "ctx", f"{t['who']} · {usd(t['value'])} through PayPal/Square · {t['n']:,} purchases in total · {t['unit']}")
        s.text(x0 + w + 8, y + 15, usd(t["value"], True), "val" if me else "lbl")
    s.line(x0, 6, x0, h - 6, "axis")
    fm_n = sum(1 for t in top if t["unit"] == "Facility Management")
    return figure(5, "Paid through PayPal or Square: more than twice any other card",
                  "Dollars paid to merchants billed through PayPal or Square, January 2018–September 2023. (FM) = Facility Management.",
                  s.done(),
                  f"{fm_n} of the top twelve cards belong to Facility Management. Other cardholders are shown under aliases: "
                  "a high value is a reason to look at receipts, not a finding about a person.",
                  table(["Cardholder", "PayPal/Square dollars", "All purchases", "Work unit"],
                        [[esc(t["who"]), usd(t["value"]), f"{t['n']:,}", esc(t["unit"])] for t in top], align=["l", "r", "r", "l"]))


def fig_scatter(R):
    pts = R["scatter"]
    h, x0, x1, y0, y1 = 360, 52, W - 20, 20, 310
    xmax, ymax = 0.85, 0.38
    s = Svg(W, h, "Each cardholder by share of spending through PayPal or Square and share of purchases just under the limit")
    s.yaxis(x0, x1, y0, y1, ymax, [0, .1, .2, .3], lambda v: pct(v))
    for v in (0, .2, .4, .6, .8):
        x = x0 + (x1 - x0) * v / xmax
        s.text(x, y1 + 18, pct(v), "tick", "middle")
    order = sorted(pts, key=lambda p: (p["who"] == "Ybarra", p["facility"]))
    for p in order:
        x = x0 + (x1 - x0) * min(p["fac"], xmax) / xmax
        y = y1 - (y1 - y0) * min(p["near"], ymax) / ymax
        me = p["who"] == "Ybarra"
        cls = "s1" if me else ("s2" if p["facility"] else "ctx")
        s.dot(x, y, 7 if me else 4.5, cls,
              f"{p['who']} · {pct(p['fac'])} of spending through PayPal/Square · {pct(p['near'], 1)} of purchases at $2,700–$3,000 · {p['n']:,} purchases")
        if me:
            s.text(x - 12, y + 4, "Ybarra", "val", "end")
    s.text((x0 + x1) / 2, h - 6, "Share of the card's spending paid through PayPal or Square", "axl", "middle")
    s.text(x0 - 40, 12, "Share of purchases at $2,700–$3,000", "axl", "start")
    rows = [[esc(p["who"]), pct(p["fac"]), pct(p["near"], 1), f"{p['n']:,}", "yes" if p["facility"] else ""]
            for p in sorted(pts, key=lambda p: -p["fac"])[:25]]
    return figure(6, "Two simple ratios separate his card from 169 others",
                  f"One dot per cardholder with at least {R['min_n']} purchases, January 2018–September 2023.",
                  s.done(),
                  "Neither ratio needs anything beyond merchant name and amount. The orange dots nearest to him are cards in his own work unit, "
                  "several of which paid the same payees (see section 7).",
                  table(["Cardholder (top 25 by PayPal/Square share)", "PayPal/Square share", "At $2,700–$3,000", "Purchases", "Facility Mgmt"], rows),
                  legend([("dotk", "s1", "Ybarra"), ("dotk", "s2", "Facility Management peers"), ("dotk", "ctx", "Other cardholders")]))


def fig_monthly(R):
    mon = R["subject_monthly"]
    h, x0, x1, y0, y1 = 300, 52, W - 12, 24, 255
    vmax = 34000
    s = Svg(W, h, "Ybarra's monthly card spending, split into PayPal or Square payments and everything else")
    s.yaxis(x0, x1, y0, y1, vmax, [0, 10000, 20000, 30000], lambda v: f"${v / 1000:.0f}K")
    band = (x1 - x0) / len(mon)
    for i, m in enumerate(mon):
        other = m["total"] - m["fac"]
        hf = (y1 - y0) * m["fac"] / vmax
        ho = (y1 - y0) * other / vmax
        tip = f"{m['month']} · PayPal/Square {usd(m['fac'])} · other {usd(other)} · total {usd(m['total'])}"
        x = x0 + band * i + 1.5
        if ho > 2:
            s.rect(x, y1 - hf, band - 3, hf, "s1", tip)
            s.col(x, y1 - hf - ho, band - 3, ho - 2, "ctx", tip, r=3)
        else:
            s.col(x, y1 - hf, band - 3, hf, "s1", tip, r=3)
        if m["month"].endswith("-01"):
            s.text(x0 + band * i, y1 + 18, m["month"][:4], "tick", "start")
    yl = y1 - (y1 - y0) * 25000 / vmax
    s.line(x0, yl, x1, yl, "ref")
    s.text(x1, yl - 6, "$25,000 monthly card limit cited in the auditor's report", "lbl", "end")
    n20 = sum(1 for m in mon if m["total"] >= 20000)
    return figure(7, "Month after month, the card ran close to its monthly limit",
                  "Ybarra's purchases by posting month, January 2018–September 2023.",
                  s.done(),
                  f"{n20} of {len(mon)} months total $20,000 or more. The monthly limit is quoted in a manager's response appended to the "
                  "auditor's report (p. 8); it is not a field in the data.",
                  table(["Month", "PayPal/Square", "Other", "Total"],
                        [[m["month"], usd(m["fac"]), usd(m["total"] - m["fac"]), usd(m["total"])] for m in mon]),
                  legend([("box", "s1", "PayPal / Square"), ("box", "ctx", "Everything else"), ("ln", "refk", "Monthly limit")]))


def fig_shared(R):
    by = [b for b in R["shared_summary"]["by_year"] if b["year"] <= 2026]
    years = list(range(2018, 2027))
    d = {b["year"]: b for b in by}
    h, x0, x1, y0, y1 = 290, 52, W - 12, 24, 245
    vmax = 290000
    s = Svg(W, h, "Payments to Ybarra's PayPal and Square payees from his card and from other cards, by year")
    s.yaxis(x0, x1, y0, y1, vmax, [0, 100000, 200000], lambda v: f"${v / 1000:.0f}K")
    band = (x1 - x0) / len(years)
    for i, yr in enumerate(years):
        b = d.get(yr, {"subject": 0, "others": 0})
        cx = x0 + band * (i + .5)
        for j, (key, cls, name) in enumerate([("subject", "s1", "Ybarra's card"), ("others", "s2", "Other cards")]):
            hh = (y1 - y0) * b[key] / vmax
            s.col(cx - 23 + j * 24, y1 - hh, 22, hh, cls, f"{yr} · {name}: {usd(b[key])}")
        s.text(cx, y1 + 18, str(yr), "tick", "middle")
    s.text(x0 + band * 7.5, y1 - 14, "$375 in the three years after he left", "lbl", "middle")
    ss = R["shared_summary"]
    return figure(8, "The same payees were paid from other cards too, and all of it stops when he leaves",
                  "Dollars paid to the 44 PayPal/Square payees on Ybarra's card, by posting year and by whose card paid.",
                  s.done(),
                  f"{ss['payees_shared']} of his {ss['payees_total']} payees were also paid from {ss['other_cardholders']} other cards, "
                  f"{usd(ss['other_dollars'])} in all, nearly all of it charged to Facility Management. "
                  "2023 covers nine months for his card. The data cannot say whether the other cards' payments were legitimate.",
                  table(["Year", "Ybarra's card", "Other cards"], [[yr, usd(d.get(yr, {}).get("subject", 0)), usd(d.get(yr, {}).get("others", 0))] for yr in years]),
                  legend([("box", "s1", "Ybarra's card"), ("box", "s2", "Other cards")]))


def rank_strip(R):
    rows_def = [("total_spend", "Largest total spending"), ("benford_fd", "Benford first digit (MAD)"),
                ("benford_fd_z", "Benford first digit, size-adjusted"), ("near_limit", "Just under the limit (count)"),
                ("facilitator", "PayPal / Square dollars"), ("weekend", "Weekend dollars"), ("composite", "Composite of eight tests")]
    tl = R["timeline"]
    head = "<tr><th class='l'>Test</th>" + "".join(f"<th>{q['quarter'][2:4]}<br>{q['quarter'][4:]}</th>" for q in tl) + "</tr>"
    body = "<tr><td class='l'>Cardholders scored</td>" + "".join(f"<td class='n'>{q['eligible']}</td>" for q in tl) + "</tr>"
    for key, label in rows_def:
        cells = []
        for q in tl:
            r = q.get(key)
            cls = "r1" if r == 1 else "r5" if r and r <= 5 else ""
            cells.append(f"<td class='{cls}'>{r if r else ''}</td>")
        body += f"<tr><td class='l'>{label}</td>{''.join(cells)}</tr>"
    return f'<div class="tw"><table class="strip"><thead>{head}</thead><tbody>{body}</tbody></table></div>'


def sustained(tl, key, k):
    """First quarter from which the rank stays at or better than k."""
    first = None
    for q in tl:
        r = q.get(key)
        if r is not None and r <= k:
            first = first or q["quarter"]
        else:
            first = None
    return first


def qname(q):
    return f"{q[:4]} Q{q[-1]}" if q else "never"


# ----------------------------------------------------------------------------- page
CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--s1:#2a78d6;--s2:#eb6834;--ctx:#cfcec6;--wash:#e6f0fc;--strong:#1c5cab;--border:rgba(11,11,11,.10);--note:#fff7e0;--noteb:#eda100}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--ctx:#4b4b47;--wash:#16304f;--strong:#3987e5;--border:rgba(255,255,255,.10);--note:#2a2410;--noteb:#c98500}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
main,header.top{max-width:860px;margin:0 auto;padding:0 20px}
header.top{padding-top:48px}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
h1{font-size:34px;line-height:1.15;margin:0 0 12px;font-weight:650;text-wrap:balance}
.dek{font-size:19px;color:var(--ink2);margin:0 0 12px;max-width:68ch}
.meta{font-size:13px;color:var(--muted);margin:0 0 28px}
h2{font-size:23px;line-height:1.25;margin:56px 0 12px;font-weight:650;text-wrap:balance}
h3{font-size:16px;margin:0 0 4px;font-weight:650}
h4{font-size:15px;margin:24px 0 6px}
p{margin:0 0 14px;max-width:72ch}
li{margin-bottom:6px;max-width:70ch}
a{color:var(--strong)}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--grid);padding:1px 5px;border-radius:4px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:8px 0 28px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .v{font-size:28px;font-weight:650;line-height:1.1}
.tile .l{font-size:13px;color:var(--ink2);margin-top:6px}
.answer{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 22px;margin:0 0 20px}
.answer ol{margin:0;padding-left:20px}
.note{background:var(--note);border-left:3px solid var(--noteb);border-radius:6px;padding:12px 16px;margin:16px 0;font-size:15px}
.note p:last-child{margin-bottom:0}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 20px;margin:22px 0}
figure{margin:0}
.sub{font-size:13px;color:var(--ink2);margin:0 0 10px}
figcaption{font-size:13.5px;color:var(--ink2);margin-top:10px;max-width:75ch}
svg.chart{display:block;width:100%;height:auto;overflow:visible}
.tick{font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums}
.axl{font-size:11.5px;fill:var(--ink2)}
.lbl{font-size:12px;fill:var(--ink2)}
.val{font-size:12px;fill:var(--ink);font-weight:600}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.ref{stroke:var(--ink);stroke-width:2}
.refline{fill:none;stroke:var(--ink);stroke-width:2;stroke-linejoin:round}
.s2line{fill:none;stroke:var(--s2);stroke-width:2;stroke-linejoin:round}
.mark{outline:none}
.mark:hover,.mark:focus{filter:brightness(1.18)}
.s1{fill:var(--s1)}.s2{fill:var(--s2)}.ctx{fill:var(--ctx)}
.dot{stroke:var(--surface);stroke-width:2}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;color:var(--ink2);margin:0 0 8px}
.key{display:inline-flex;align-items:center;gap:6px}
.sw.box{width:11px;height:11px;border-radius:3px;background:var(--ctx)}
.sw.dotk{width:10px;height:10px;border-radius:50%;background:var(--ctx)}
.sw.ln{width:16px;height:2px;background:var(--ink)}
.sw.s1{background:var(--s1)}.sw.s2,.sw.s2k{background:var(--s2)}.sw.ctx{background:var(--ctx)}.sw.refk{background:var(--ink)}
details{margin:10px 0 14px;font-size:13px}
summary{cursor:pointer;color:var(--ink2)}
.tw{overflow-x:auto;margin:10px 0 16px}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
th{font-weight:600;color:var(--ink2);font-size:12.5px;text-align:right;white-space:nowrap}
td.r,th.r{text-align:right}td.l,th.l{text-align:left}
td.hl{font-weight:650}
tr.me td{background:var(--wash);font-weight:600}
table.tight td.l{white-space:nowrap}
table.strip{font-size:12px;min-width:820px}
table.strip th{font-size:10.5px;line-height:1.2;padding:4px 2px;text-align:center}
table.strip td{text-align:center;padding:5px 2px;border-bottom:2px solid var(--surface);border-right:2px solid var(--surface)}
table.strip td.l{text-align:left;white-space:nowrap;padding-right:10px}
table.strip td.n{color:var(--muted)}
table.strip td.r1{background:var(--strong);color:#fff;font-weight:650}
table.strip td.r5{background:var(--wash)}
.pill{display:inline-block;font-size:11.5px;padding:1px 8px;border-radius:999px;border:1px solid var(--border);color:var(--ink2);white-space:nowrap}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--page);font-size:12.5px;line-height:1.4;padding:7px 10px;border-radius:6px;
max-width:300px;opacity:0;transition:opacity .08s;z-index:10}
footer{max-width:860px;margin:56px auto 40px;padding:16px 20px 0;border-top:1px solid var(--grid);font-size:13px;color:var(--muted)}
@media print{.card,.tile,.answer{break-inside:avoid}#tip{display:none}}
"""

JS = """
const tip=document.getElementById('tip');
function show(e){const t=e.target.closest('[data-tip]');if(!t){tip.style.opacity=0;return}
 tip.textContent='';t.dataset.tip.split(' \\u00b7 ').forEach((s,i)=>{const d=document.createElement('div');d.textContent=s;if(i==0)d.style.fontWeight='650';tip.appendChild(d)});
 const r=t.getBoundingClientRect();let x=(e.clientX||r.left+r.width/2)+14,y=(e.clientY||r.top)+14;
 tip.style.opacity=1;const w=tip.offsetWidth,h=tip.offsetHeight;if(x+w>innerWidth-8)x=innerWidth-w-8;if(y+h>innerHeight-8)y=y-h-28;
 tip.style.left=x+'px';tip.style.top=y+'px'}
document.addEventListener('pointermove',show);document.addEventListener('focusin',show);
document.addEventListener('focusout',()=>tip.style.opacity=0);document.addEventListener('pointerleave',()=>tip.style.opacity=0);
"""


def build(R, issues, scores):
    o, sp, ss, lg = R["overview"], R["subject_split"], R["shared_summary"], R["legacy"]
    T = {t["key"]: t for t in R["tests"]}
    N = R["eligible"]
    tl = R["timeline"]
    pops = R["benford_pops"]
    n_fail = int((scores["fd_mad"] > 0.015).sum())
    mgr = {m["manager"]: m for m in R["subject_fac_by_manager"]}
    am = {m["manager"]: m for m in R["approvals"]["subject_managers"]}
    q3, q4 = R["queue"][3], R["queue"][4]
    fac_fm = R["fac_by_unit"][0]
    S = []
    A = S.append

    A(f"""<header class="top"><p class="eyebrow">Austin Energy purchasing cards · analysis for case-study design</p>
<h1>Could the transaction data alone have caught it?</h1>
<p class="dek">Benford's Law and ten other standard audit tests, run on Austin Energy's purchasing-card export, to see which
would have put Mark Ybarra's card on a review list, and how early.</p>
<p class="meta">Prepared November 14, 2026 · Data: City of Austin public information requests C340726-102326 and C343310-110726 ·
Reference: Austin City Auditor, <i>Austin Energy Fraud and Waste</i>, investigation IN24001, December 2025</p></header><main>""")

    A(f"""<div class="tiles">
<div class="tile"><div class="v">5th of {N}</div><div class="l">Ybarra's rank on the plain Benford first-digit test</div></div>
<div class="tile"><div class="v">1st of {N}</div><div class="l">His rank on dollars paid through PayPal or Square, in every quarter from 2018 Q2</div></div>
<div class="tile"><div class="v">{usd(sp['fac_dollars'], True)}</div><div class="l">Paid from his card through PayPal or Square; the auditor found about $980,000 in fraudulent transactions</div></div>
<div class="tile"><div class="v">{pct(sp['near_n'] / o['subject_purchases'])}</div><div class="l">Of his purchases were within 10% below the $3,000 purchase limit; the median cardholder has {pct(T['near_limit_share']['median'], 0)}</div></div>
</div>

<div class="answer"><ol>
<li><b>Benford's Law flags the card, but not decisively on its own.</b> Ranked by first-digit deviation he is 5th of {N} cardholders,
behind four small cards with ordinary explanations. Adjusted for sample size he is 1st. {pct(pops['Ybarra']['fd'][1])} of his purchases start with a 2 against an expected 17.6%.</li>
<li><b>Benford's real contribution is where it points.</b> The excess is on amounts starting 28 and 29, that is, just under $3,000,
and the amounts in the data show $3,000 was the single-purchase limit.</li>
<li><b>The strongest single test is the payment channel.</b> {usd(sp['fac_dollars'])} of his {usd(o['subject_spend'])} went to payees billed through
PayPal or Square, more than twice any other card. He ranks first on that test in every quarter it can be computed, and was
already first of {lg['cardholders']} cardholders in his first three statements in the older system (2017).</li>
<li><b>A simple combination of textbook tests ranks him 1st</b>, and stays in the top three from mid-2019, four years before the referral to the auditor in October 2023.</li>
<li><b>Two cautions for the case study.</b> The release has no fraud labels, so nothing here measures precision or recall. And
{ss['payees_shared']} of his {ss['payees_total']} PayPal/Square payees were also paid from {ss['other_cardholders']} other cards in his work unit
({usd(ss['other_dollars'])}); the data cannot tell which of those payments were legitimate.</li>
</ol></div>""")

    # ------------------------------------------------------------------ 1 data
    fixed = issues[issues.action.str.startswith("fixed")]
    A(f"""<h2 id="data">1. The combined data, and one problem that changes every total</h2>
<p>The two releases were combined into one file, <code>data/ae_pcard_combined.csv</code>: {151344:,} rows, every row the City released,
in one set of columns. It holds {30744:,} accounting lines from the older system (fiscal years 2013–2018, no dates) and {120600:,} lines
from the newer one (January 2018–September 2026), of which {7959:,} come from the second request and fill the January–June 2019 gap.
A second file, <code>data/ae_pcard_transactions.csv</code>, has one row per card charge for the newer system ({o['works_transactions']:,} charges).</p>
<div class="note"><p><b>The "Allocated Amount" column is not an allocation.</b> Every one of the 1,716 transactions that spans more than one line carries
the same amount on every line: a $103.18 grocery purchase appears on 47 item lines, each showing $103.18. The column is the transaction total, repeated.
Adding it up overstates newer-system spending by $2.04 million ($65.95M instead of $63.91M). The combined file keeps the amounts as released and adds
<code>is_primary_line</code> so that totals can be taken once per transaction.</p></div>
<p>{len(issues)} issues are logged in <code>data/data_quality_issues.csv</code>. {len(fixed)} were repaired; the rest are flagged or documented because the right value cannot be known.
No row was dropped and no amount was changed. The main repairs:</p>""")
    key_ids = ["W01", "W02", "W17", "W16", "W03", "W10", "W13", "W15", "W04", "W18", "L01", "L03", "L05"]
    rows = []
    for iid in key_ids:
        r = issues[issues.issue_id == iid].iloc[0]
        rows.append([esc(r.issue), f"{int(r.rows_affected):,}" if r.rows_affected else "–", f'<span class="pill">{esc(r.action)}</span>', esc(r.detail)])
    A(table(["Issue", "Rows", "Action", "What was done"], rows, align=["l", "r", "l", "l"]))
    allrows = [[r.issue_id, esc(r.field), esc(r.issue), f"{int(r.rows_affected):,}", esc(r.action)] for r in issues.itertuples()]
    A("<details><summary>All logged issues</summary>" + table(["ID", "Field", "Issue", "Rows", "Action"], allrows, align=["l", "l", "l", "r", "l"]) + "</details>")
    A("""<p>Two things could not be fixed and are worth a follow-up request. The older sheet ends with the December 2017 accounting period and the newer
one starts with charges posted on January 2, 2018, so roughly one statement cycle (purchases posted in December 2017) is in neither. And the older
sheet has no dates at all, only fiscal periods.</p>""")

    # ------------------------------------------------------------------ 2 benford
    f2 = T["benford_f2"]
    above = R["top"]["benford_fd"][:4]
    A(f"""<h2 id="benford">2. Benford's Law: does it flag him?</h2>
<p>The test uses one amount per card charge, positive amounts only, for the {o['period_transactions']:,} transactions posted from January 2018
through September 2023 (his last purchase was September 1, 2023). Cardholders need at least {R['min_n']} purchases to be scored, which leaves {N}.</p>""")
    A(fig_first_digit(R))
    rows = []
    for k in ["Ybarra", "Facility Management peers", "All other cardholders", "All cardholders, Oct 2023-Sep 2026", "Legacy system, FY2013-FY2018 (accounting lines)"]:
        v = pops[k]
        rows.append([esc(k.replace("-", "–")), f"{v['n']:,}", pct(v["fd"][1], 1), f"{v['fd_mad']:.4f}", f"{v['f2_mad']:.4f}"])
    A(table(["Population", "Purchases", "Start with 2", "First-digit MAD", "First-two-digit MAD"], rows))
    A(f"""<p>MAD is the mean absolute deviation between observed and expected digit shares. Nigrini's widely used guideline treats a first-digit MAD above 0.015
as nonconforming. The department as a whole sits at about 0.011, which is acceptable; Ybarra is at {pops['Ybarra']['fd_mad']:.3f}. His chi-square statistic is
{pops['Ybarra']['fd_chi2']:.0f} on eight degrees of freedom.</p>
<p>That sounds conclusive, but it is not a usable rule at the level of a single card: <b>{n_fail} of the {N} cardholders exceed 0.015</b>.
An individual's card is a small, lumpy sample (one person buying the same things at the same prices), so nearly everyone "fails". What matters is the ranking.</p>""")
    rows = [[f"{b['min_n']}", b["eligible"], b["fd_mad"], b["fd_z"], b["f2_mad"], b["f2_z"]] for b in R["benford_min_n"]]
    A("<h4>Ybarra's Benford rank, by test and by the minimum number of purchases required</h4>" +
      table(["Minimum purchases", "Scored", "First digit", "First digit, adjusted", "Two digits", "Two digits, adjusted"], rows))
    A(f"""<p>On the plain first-digit MAD he is {T['benford_fd']['rank']}th of {N}. The four cards above him are small ({', '.join(str(a['n']) for a in above)} purchases)
and dominated by one vendor or one repeated price: on those four, a single vendor accounts for {pct(min(a['top_vendor_share'] for a in above))}–{pct(max(a['top_vendor_share'] for a in above))} of purchases.
MAD is larger for small samples even when nothing is wrong, so the ranking favors small cards. The size-adjusted version compares each card's MAD with what
random samples of the same size from Benford's distribution produce; on that measure he is {T['benford_fd_z']['rank']}st, by a wide margin.</p>
<p>The first-two-digit test, often recommended as the more powerful one, does worse here in its plain form: rank {f2['rank']} of {N}. With 90 digit
categories and a few hundred purchases per card, noise swamps the signal. It is still the test that shows <i>where</i> the problem is.</p>""")
    A(fig_two_digit(R))
    A(f"""<p><b>Verdict on Benford.</b> It would have put his card on a list of five to review, and from early 2021 it does so in every quarter
(section 6). It would not have singled him out, it produces obvious false alarms above him, and its statistical significance is meaningless at card level.
Its best use in this case is diagnostic: the 28/29 spike turns "something is off with the digits" into a specific question about purchases just under $3,000.</p>""")

    # ------------------------------------------------------------------ 3 limit
    A("""<h2 id="limit">3. Where the digits point: the purchase limit</h2>""")
    A(fig_limit(R))
    A(fig_hist(R))
    rows = [["Purchases", f"{sp['fac_n']:,}", f"{sp['other_n']:,}"],
            ["Dollars", usd(sp["fac_dollars"]), usd(sp["other_dollars"])],
            ["Median amount", usd(sp["fac_median"]), usd(sp["other_median"])],
            ["At $2,700–$3,000", pct(sp["fac_near_limit_share"]), pct(sp["other_near_limit_share"])],
            ["Whole-dollar amounts", pct(sp["fac_whole_dollar_share"]), pct(sp["other_whole_dollar_share"]) + " (of purchases ≥ $100)"],
            ["Bought on a weekend", pct(sp["fac_weekend_share"]), pct(sp["other_weekend_share"])]]
    A("<h4>His card splits cleanly into two populations</h4>" + table(["", "Paid through PayPal / Square", "Everything else on his card"], rows))
    A(f"""<p>The second column looks like a maintenance worker's card: Home Depot, plumbing suppliers, a median purchase of {usd(sp['other_median'])}.
The first column is {sp['fac_n']} payments with a median of {usd(sp['fac_median'])}, {pct(sp['fac_whole_dollar_share'])} of them in whole dollars.
Real invoices for labor and materials rarely come out at exactly $2,900; {sp['exact_2900']} of his did.
On {R['subject_stacked']['days']} days he made two or more just-under-limit purchases on the same day, up to {R['subject_stacked']['max_n']}
({usd(R['subject_stacked']['max_dollars'])} on {R['subject_stacked']['max_day']}, a Saturday).</p>""")

    # ------------------------------------------------------------------ 4 other tests
    A(f"""<h2 id="tests">4. Other tests, and where each one ranks him</h2>
<p>All of these are standard purchasing-card audit tests, and each needs only fields that are in the export. "Rank" is his position among the {N}
cardholders with at least {R['min_n']} purchases, and among the {R['eligible_facility']} of them in Facility Management.</p>""")
    needs = {"total_spend": "amount", "benford_fd": "amount", "benford_fd_z": "amount", "benford_f2": "amount", "benford_f2_z": "amount",
             "near_limit": "amount, limit", "near_limit_share": "amount, limit", "whole_dollar": "amount", "facilitator": "merchant name",
             "sole_vendor": "merchant name", "split": "merchant, date, amount", "stacked": "date, amount", "weekend": "purchase date",
             "repeat": "amount", "mismatch": "merchant category, account code", "months": "post date, amount",
             "composite": "all of the above", "iforest": "all of the above", "lof": "all of the above"}

    def fmtv(t, v):
        m = t["metric"]
        if "dollars" in m or m == "spend":
            return usd(v, True)
        if "share" in m:
            return pct(v, 0 if v > .1 else 1)
        if m in ("fd_mad", "f2_mad"):
            return f"{v:.3f}"
        if m in ("composite", "iforest", "lof", "fd_mad_z_benford", "f2_mad_z"):
            return f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
        return f"{v:,.0f}"
    rows = []
    for t in R["tests"]:
        rows.append([esc(t["label"]), f"<b>{t['rank']}</b>" if t["rank"] <= 5 else str(t["rank"]), str(t["rank_facility"]),
                     fmtv(t, t["value"]), fmtv(t, t["median"]), esc(needs[t["key"]])])
    A(table(["Test", f"Rank of {N}", f"Rank of {R['eligible_facility']} in unit", "Ybarra", "Median cardholder", "Fields needed"], rows, align=["l", "r", "r", "r", "r", "l"]))
    A(fig_fac_rank(R))
    A(fig_scatter(R))
    A(f"""<p><b>What worked.</b> Payment channel, just-under-limit amounts, weekend spending, and same-day stacking each rank him first. They work because they
describe <i>how</i> a fictitious-vendor scheme has to operate on a card: the payee must be able to accept a card without being a real merchant
(a PayPal or Square account does that), and each charge must stay under the limit that would trigger a second look.</p>
<p><b>What did not.</b> Three tests that textbooks recommend rank him poorly, and the reasons are instructive:</p>
<ul>
<li><b>Whole-dollar amounts</b> (rank {T['whole_dollar']['rank']}): {pct(sp['fac_whole_dollar_share'])} of his PayPal/Square payments are whole dollars, but the 590 routine purchases on the
same card dilute the ratio. Tests run on a whole card can hide a pattern that is obvious on a subset.</li>
<li><b>Split purchases</b> (rank {T['split']['rank']}): he rarely paid the <i>same</i> payee twice within a week. He spread near-limit payments across many payees,
which defeats the usual same-vendor split test. The "same-day stacking" variant, which ignores the payee, ranks him first.</li>
<li><b>Payees no one else uses</b> (rank {T['sole_vendor']['rank']}, but only {usd(T['sole_vendor']['value'], True)}): most of his payees were <i>not</i> exclusive to his card. See section 7.</li>
</ul>
<p><b>Machine-learning methods.</b> An isolation forest and a local-outlier-factor model, given ten ratios per cardholder and no labels, both rank him first.
They add little here beyond the composite score, which is easier to explain to a manager: it is the average of his percentile rank on eight tests.
He ranks first on four of them.</p>
<p><b>The naive baseline matters.</b> Simply sorting by total spending puts him {T['total_spend']['rank']}th. The four cards above him are high-volume inventory
and operations buyers. Any method worth teaching should beat "look at the biggest spenders", and plain Benford (also 5th) does not.</p>""")

    # ------------------------------------------------------------------ 5 payees
    A(f"""<h2 id="payees">5. The payees</h2>
<p>His PayPal/Square payments went to {sp['fac_payees']} payees. Payee names are replaced with aliases here because the auditor's report does not name them
and the data cannot show which ones the auditor counted as fictitious; the descriptors are in the data files as the City released them.</p>""")
    rows = [[esc(p["descriptor"]), p["n"], usd(p["dollars"]), usd(p["median"]), f"{p['first'][:4]}–{p['last'][:4]}", esc(p["mcc"].title()), p["other_users"]]
            for p in R["subject_payees"][:15]]
    A(table(["Payee", "Payments", "Dollars", "Median", "Active", "Merchant category (card network)", "Other cards"], rows,
            align=["l", "r", "r", "r", "l", "l", "r"], cls="tight"))
    mcc = R["subject_mcc"]
    A(f"""<p>Two things stand out without opening a single invoice. First, the medians: fifteen different payees, all billing a median between $2,650 and $2,900.
Second, the merchant categories. {usd(R['subject_object'][0]['dollars'])} of these payments were coded by Ybarra to "maintenance-buildings", but the card network
classified the payees as {esc(mcc[0]['mcc'].title())} ({usd(mcc[0]['dollars'], True)}), {esc(mcc[1]['mcc'].title())} ({usd(mcc[1]['dollars'], True)}),
{esc(mcc[2]['mcc'].title())} ({usd(mcc[2]['dollars'], True)}), {esc(mcc[3]['mcc'].title())} ({usd(mcc[3]['dollars'], True)}),
{esc(mcc[4]['mcc'].title())} ({usd(mcc[4]['dollars'], True)}) and {esc(mcc[5]['mcc'].title())} ({usd(mcc[5]['dollars'], True)}).
A PayPal or Square account holder picks their own category, so the category says little about the work, and that is the point:
these are not merchants a card network has vetted.</p>
<p>The automated version of this check (flag purchases whose merchant category rarely goes with that account code for anyone else) ranks him only
{T['mismatch']['rank']}th, because "general contractor" coded to building maintenance is not unusual. The mismatch is visible to a reader, less so to a rule.</p>""")
    A(f"""<p><b>How this lines up with the auditor's figures.</b> The auditor reports about $980,000 paid to fictitious vendors in fiscal years 2018–2023, about $750,000
approved by his first supervisor and about $170,000 by an interim one. In the data, his PayPal/Square payments total {usd(sp['fac_dollars'])}
(plus {usd(lg['subject_fac'])} in the older system), of which {usd(mgr['MGR-194']['dollars'])} was signed off by manager ID MGR-194
({mgr['MGR-194']['first'][:7]} to {mgr['MGR-194']['last'][:7]}) and {usd(mgr['MGR-040']['dollars'])} by MGR-040 ({mgr['MGR-040']['first'][:7]} to {mgr['MGR-040']['last'][:7]}).
The pattern and the dates match. The overall total and the first approver's total are within 5% of the auditor's figures; the second is about 29% higher. "Paid through PayPal or Square" is therefore a reasonable stand-in for the auditor's list, not a copy of it.</p>""")

    # ------------------------------------------------------------------ 6 timing
    fq = {k: sustained(tl, k, 5) for k in ["benford_fd", "benford_fd_z", "composite", "near_limit", "facilitator"]}
    c3 = sustained(tl, "composite", 3)
    A(f"""<h2 id="timing">6. How early?</h2>
<p>Each column below uses only transactions posted up to the end of that quarter, and re-ranks every cardholder who had reached {R['min_n']} purchases by then.
Dark cells are rank 1; shaded cells are ranks 2–5.</p>""")
    A(rank_strip(R))
    A(f"""<p>The payment-channel test has him first in every quarter. The near-limit count never leaves the top three. The composite stays in the top three from {qname(c3)}. Plain Benford is erratic for three years (as low as 15th in early 2019, when many newly eligible small cards
enter the ranking) and only settles in the top five from {qname(fq['benford_fd'])}; the size-adjusted version gets there by {qname(fq['benford_fd_z'])}.</p>
<p><b>Even earlier.</b> The older system has no dates or merchant categories, but it has merchant names. His card first appears there in the last accounting period
of fiscal year 2017. Across his first three statements he has {lg['subject_lines']} purchases totalling {usd(lg['subject_spend'])}, of which {usd(lg['subject_fac'])}
({lg['subject_fac_n']} payments) went through PayPal or Square: first of the {lg['cardholders']} cardholders active in those periods, and a third of everything the department
paid through those two processors. The pattern was there from the first statement.</p>""")
    A(fig_monthly(R))
    rows = [[a["fy"], a["eligible"], a["subject_n"], a["total_spend"], a["benford_fd"], a["benford_fd_z"], a["near_limit"], a["facilitator"], a["weekend"]] for a in R["annual"]]
    A("<h4>If the review were done once a year, on that fiscal year's transactions only (cardholders with at least 50 purchases)</h4>" +
      table(["Fiscal year", "Scored", "His purchases", "Spending", "Benford", "Benford, adjusted", "Near limit", "PayPal / Square", "Weekend"], rows))
    A("""<div class="note"><p><b>What "would have caught it" can and cannot mean.</b> These are rankings rebuilt from a 2026 extract, with tests chosen by someone who knows how
the story ends. They show the pattern was visible in the data; they do not show that a reviewer in 2019 would have run these tests or acted on them. The tests are
standard ones, though, and the two strongest need nothing more than a sort on merchant name and amount.</p></div>""")

    # ------------------------------------------------------------------ 7 shared payees
    A(f"""<h2 id="shared">7. The same payees on other cards</h2>
<p>This is the finding I would be most careful with in a classroom. The auditor's report is about one card. But {ss['payees_shared']} of the {ss['payees_total']} PayPal/Square payees on
Ybarra's card were also paid from {ss['other_cardholders']} other cards, {ss['other_n']} payments totalling {usd(ss['other_dollars'])}, and all but $2,005 of that was charged to
Facility Management. Those payments have a similar shape ({pct(ss['other_whole_dollar_share'])} whole-dollar, {pct(ss['other_near_limit_share'])} within 10% below the limit).
After September 2023 the same payees receive one payment of {usd(ss['after_dollars'])} in three years.</p>""")
    A(fig_shared(R))
    rows = [[esc(u["unit"]), u["cardholders"], usd(u["fac"]), usd(u["spend"]), pct(u["fac"] / u["spend"])] for u in R["fac_by_unit"][:6]]
    A("<h4>PayPal/Square spending by work unit, January 2018–September 2023</h4>" +
      table(["Work unit (cardholder's usual unit)", "Cardholders", "PayPal / Square", "All purchases", "Share"], rows))
    A(f"""<p>Facility Management accounts for {usd(fac_fm['fac'], True)} of the department's {usd(R['fac_total_period'], True)} in PayPal/Square spending, {pct(fac_fm['fac'] / fac_fm['spend'])} of everything the unit
bought by card. A unit-level view would have raised the question before any single card did.</p>
<p>There are several possible readings and the data cannot choose between them: the payees could be real small contractors that the whole facilities group used
(which would complicate the word "fictitious"); colleagues could have paid them at Ybarra's request or on his recommendation; or the problem could be wider than one card.
The auditor's report says staff interviewed "were largely unaware of the vendors that Ybarra had utilized", which sits uneasily with these payments.
Nothing here is a finding about any other cardholder, which is why they appear only as aliases. For the case study it means two things:
treating "Ybarra's card" as the only positive label is probably too narrow, and vendor-exclusivity tests fail for an interesting reason.</p>""")

    # ------------------------------------------------------------------ 8 workload
    rows = []
    for q in R["queue"][1:]:
        rows.append([f"{q['min_flags']} or more", f"{q['n']:,}", q["cardholders"], f"{q['subject_n']:,}", pct(q["subject_n"] / q["n"]),
                     pct(q["subject_fac_dollars"] / sp["fac_dollars"])])
    A(f"""<h2 id="workload">8. What a review queue would look like</h2>
<p>Ranking cardholders tells a manager whose statement to pull. A transaction-level queue tells a reviewer which receipts to ask for. Each purchase gets one point for each of
five simple flags: amount at 90–100% of the $3,000 limit; whole-dollar amount; paid through PayPal or Square; payee used by no other cardholder; purchased on a weekend.</p>""" +
      table(["Flags", "Purchases in queue", "Cardholders", "Ybarra's", "His share of queue", "His PayPal/Square $ captured"], rows) +
      f"""<p>Asking for receipts on every purchase with three or more flags means {q3['n']:,} purchases over almost six years, about {q3['n'] / 69:.0f} a month across the whole department.
One in five would be his, and the queue would contain {pct(q3['subject_fac_dollars'] / sp['fac_dollars'])} of his PayPal/Square dollars. At four or more flags it is {q4['n']} purchases
and more than half are his. These shares describe his card, not confirmed fraud, and the other purchases in the queue are unlabelled.</p>""")

    # ------------------------------------------------------------------ 9 approvals
    m194, m040 = am.get("MGR-194", {}), am.get("MGR-040", {})
    A(f"""<h2 id="approvals">9. What the sign-off timestamps show</h2>
<p>The export has cardholder and manager sign-off timestamps and an anonymized manager ID. Every one of his purchases through July 2023 has a manager sign-off;
{R['approvals']['subject_no_signoff']} from August–September 2023 have none, which fits the report's account of management starting to ask questions.
Across the {R['approvals']['managers']} manager IDs with at least 200 sign-offs, the typical manager records {pct(R['approvals']['median_within_10s'])} of sign-offs within ten seconds of
their previous one: approval is mostly done as a batch. MGR-040 does the same ({pct(m040.get('within_10s', 0))}). MGR-194, who signed off {m194.get('n_subject', 0)} of his transactions,
is the opposite: only {pct(m194.get('within_10s', 0))} within ten seconds, with a median gap of {m194.get('median_gap', 0):.0f} seconds between sign-offs.</p>
<p>Neither pattern shows whether an invoice was opened, and a timestamp cannot measure attention. But it does support the auditor's second finding in a way students can
verify: sign-off was recorded for every transaction for more than five years. The control existed and ran; it did not test anything.</p>""")

    # ------------------------------------------------------------------ 10 comparison
    rows = [
        ["Row counts, 2019 gap, 287 repeated lines, 1,237 swapped sign-off cells", "Same counts", "Same counts", "Agree"],
        ["Benford first-digit rank (≥100 purchases)", "5 of 170", f"{T['benford_fd']['rank']} of {N}", "Agree"],
        ["Benford first-two-digit rank", "52 of 170", f"{T['benford_f2']['rank']} of {N}", "Agree (one place apart, from the amount treatment below)"],
        ["Share of his purchases starting with 2; purchases of exactly $2,900", "50.2%; 77", f"{pct(pops['Ybarra']['fd'][1], 1)}; {sp['exact_2900']}", "Agree"],
        ["Transaction amount", "Sum of allocation lines", "One amount per transaction",
         "<b>Differ.</b> The lines repeat the transaction total, so summing overstates. Newer-system net spending: $65.95M (ChatGPT) vs $63.91M."],
        ["Ybarra's purchases", "$1,335,366.74", usd(o["subject_spend"]) + ".61"[0:0], "Differ slightly, same cause (one two-line transaction)"],
        ["Swapped sign-off cells", "Left as released, dates set to unknown", "Swapped back", "Differ: the block is contiguous and the repair is checkable"],
        ["Cardholder names, code formats, country codes, blank descriptions", "Left as released", "Standardized, originals kept",
         "Differ. 47 people appear under two spellings, mostly in the last three months of the release, which splits their history if left alone"],
        ["Older system (2013–2018)", "Appended with its own columns", "Mapped into the same columns; merchant and description separated",
         "Differ. Mapping it shows his first statements already had the pattern"],
        ["Purchase limit", "\"Confirm applicable limits\"", "Inferred from the amount distribution: $3,000, with a second cliff at $5,000 from 2023", "Extends"],
        ["Tests beyond Benford", "None", "Ten more, a composite, two unsupervised models, a review queue", "Extends"],
        ["Other cards paying the same payees", "Not examined", f"{ss['other_cardholders']} cards, {usd(ss['other_dollars'])}", "New"],
    ]
    rows[5][2] = f"${o['subject_spend']:,.2f}"
    A("""<h2 id="compare">10. How this compares with the ChatGPT report</h2>
<p>The ChatGPT report covers Benford carefully and its Benford numbers reproduce. The differences are in data preparation and in scope.</p>""" +
      table(["Item", "ChatGPT report", "This analysis", "Assessment"], rows, align=["l", "l", "l", "l"]))

    # ------------------------------------------------------------------ 11 limits + methods
    A(f"""<h2 id="limits">11. Limits, and what to ask the City for</h2>
<ul>
<li><b>No labels.</b> The release does not say which transactions the auditor counted in the $980,000. A schedule of those transactions would turn this from a ranking exercise into a
detection benchmark with measurable precision and recall.</li>
<li><b>The purchase limit is inferred.</b> Ask for the single-purchase and monthly limits and the dates they changed.</li>
<li><b>Amount semantics.</b> Ask the City to confirm that "Allocated Amount" is the transaction total, and whether true allocation amounts exist.</li>
<li><b>December 2017.</b> One statement cycle appears to be missing between the two systems.</li>
<li><b>Fiscal periods in the older sheet.</b> Period 1 never appears and period 12 is double-sized; a confirmed period-to-month mapping would allow dating those rows.</li>
<li><b>Legal status.</b> The auditor's report says Ybarra was indicted and arrested in September 2025. That is not a conviction, and I have not checked the outcome of the criminal case.</li>
<li><b>Payment-processor detection is by descriptor.</b> A merchant is treated as PayPal/Square when its name starts with PAYPAL, PP, SQ or SQU, or the card network lists PayPal or Square/Block as the
processor. Some legitimate small businesses bill this way; the test measures a channel, not intent.</li>
</ul>
<h2 id="methods">12. Methods and files</h2>
<ul>
<li><b>Population.</b> Newer-system card charges posted {R['period'][0]} to {R['period'][1]}; positive amounts for all digit and amount tests; credits kept in the data.
Split charges (IDs ending -A/-B/-C) are added back together.</li>
<li><b>Benford.</b> First digit and first two digits on integer cents. MAD against log10(1 + 1/d). Size-adjusted score: (MAD − mean) / standard deviation of the MAD of 400
random samples of the same size drawn from Benford's distribution.</li>
<li><b>Just under the limit.</b> $2,700.00–$3,000.00 (90–100% of the inferred limit).</li>
<li><b>Split purchases.</b> Same cardholder and payee, two purchases within 7 days, each at or under the limit, together over it. Same-day stacking: two or more near-limit purchases on one day, any payee.</li>
<li><b>Payee key.</b> Merchant name with processor prefix, digits and punctuation removed, first 12 letters. This merges descriptor variants of one payee and can occasionally merge two different ones.</li>
<li><b>Category mismatch.</b> A purchase is flagged when fewer than 2% of other cardholders' purchases in that merchant category use the same account (object) code.</li>
<li><b>Composite.</b> Mean percentile rank on eight tests: size-adjusted Benford, near-limit count, whole-dollar share, PayPal/Square dollars, sole-payee dollars, split purchases, weekend dollars, category mismatch.</li>
<li><b>Unsupervised models.</b> Isolation forest (500 trees) and local outlier factor (20 neighbors) on ten standardized cardholder ratios.</li>
<li><b>Aliases.</b> Cardholders other than Ybarra are CH-001, CH-002, … in order of total spending; payees are numbered. Run <code>scripts/analyze.py --show-names</code> to see the released names.</li>
</ul>
<p>Files: <code>data/ae_pcard_combined.csv</code>, <code>data/ae_pcard_transactions.csv</code>, <code>data/data_quality_issues.csv</code>, <code>data/README.md</code> (column dictionary),
<code>analysis/cardholder_scores.csv</code>, <code>analysis/subject_transactions.csv</code>, <code>analysis/results.json</code>.
To reproduce: <code>scripts/build_combined.py</code>, then <code>scripts/analyze.py</code>, then <code>scripts/build_report.py</code>.</p>
</main><footer>Rankings identify cards worth a closer look. They are not findings about any person other than the subject of the auditor's published report.</footer>
<div id="tip" role="status"></div>""")

    body = "".join(S)
    return ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Could the transaction data alone have caught it? Austin Energy ProCard analysis</title><style>" + CSS + "</style></head><body>"
            + body + "<script>" + JS + "</script></body></html>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="analysis")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="reports/austin_energy_pcard_fraud_detection.html")
    a = ap.parse_args()
    R = json.loads((Path(a.analysis) / "results.json").read_text())
    issues = pd.read_csv(Path(a.data) / "data_quality_issues.csv").fillna("")
    scores = pd.read_csv(Path(a.analysis) / "cardholder_scores.csv")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(R, issues, scores), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
