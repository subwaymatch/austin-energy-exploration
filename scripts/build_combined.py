#!/usr/bin/env python3
"""Build the combined, cleaned Austin Energy ProCard dataset.

Inputs (City of Austin public information requests):
  --original    C340726_AE.xlsx   (sheets "PIR - MS Data 2013-2018", "PIR - Works Data 2018-2026")
  --supplement  C343310.xlsx      (sheet "Sheet1": Works rows posted January-June 2019)

Outputs (written to --out, default ./data):
  ae_pcard_combined.csv        every released row, one harmonized schema (line level)
  ae_pcard_transactions.csv    derived: one row per card charge, Works system only
  data_quality_issues.csv      every issue found, rows affected, and what was done
  cardholder_name_map.csv      released name -> standardized name

Every repair is logged in data_quality_issues.csv and marked per row in `dq_flags`.
No row is dropped; no amount is changed.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

LEGACY_SHEET = "PIR - MS Data 2013-2018"
WORKS_SHEET = "PIR - Works Data 2018-2026"
SUPP_SHEET = "Sheet1"
# File names as released by the City (request C340726-102326 and follow-up C343310-110726)
ORIGINAL_NAME = "C340726_AE.xlsx"
SUPPLEMENT_NAME = "C343310.xlsx"

ISSUES = []


def log(issue_id, source, field, issue, rows, action, detail=""):
    ISSUES.append(dict(issue_id=issue_id, source=source, field=field, issue=issue,
                       rows_affected=int(rows), action=action, detail=detail))


def add_flag(df, mask, flag):
    cur = df.loc[mask, "dq_flags"]
    df.loc[mask, "dq_flags"] = np.where(cur == "", flag, cur + ";" + flag)


def squash(s):
    """Trim, and collapse runs of whitespace (incl. newlines/tabs) to one space."""
    return s.where(s.isna(), s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip())


def code4(s):
    """Accounting codes arrive as a mix of ints and strings: store as text."""
    def one(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        if isinstance(x, (int, float, np.integer, np.floating)):
            return str(int(x))
        return str(x).strip()
    return pd.Series([one(x) for x in s], index=s.index, dtype=object)


# --------------------------------------------------------------------------- Works
def load_works(original, supplement):
    w1 = pd.read_excel(original, sheet_name=WORKS_SHEET, dtype=object)
    w2 = pd.read_excel(supplement, sheet_name=SUPP_SHEET, dtype=object)
    assert list(w1.columns) == list(w2.columns), "Works sheets must share a schema"
    w1["source_file"], w1["source_sheet"] = ORIGINAL_NAME, WORKS_SHEET
    w2["source_file"], w2["source_sheet"] = SUPPLEMENT_NAME, SUPP_SHEET
    for d in (w1, w2):
        d["source_row"] = np.arange(len(d)) + 2  # Excel row number (header is row 1)
    overlap = set(w1["Txn ID"]) & set(w2["Txn ID"])
    assert not overlap, f"{len(overlap)} transaction IDs appear in both releases"
    log("W01", "works", "Post Date",
        "Original release has no rows posted January-June 2019", len(w2), "fixed",
        f"Supplemental release C343310 adds {len(w2):,} rows / {w2['Txn ID'].nunique():,} transaction IDs; "
        "no transaction ID appears in both releases. Every posting month January 2018-September 2026 now has rows.")
    w = pd.concat([w1, w2], ignore_index=True)
    w["dq_flags"] = ""
    return w


def clean_works(w):
    released_cols = [c for c in w.columns if c not in ("source_file", "source_sheet", "source_row", "dq_flags")]
    as_released = w[released_cols].astype(str)
    w["is_exact_duplicate_line"] = as_released.duplicated(keep="first")
    n_dup_txns = w.loc[as_released.duplicated(keep=False), "Txn ID"].nunique()
    del as_released

    # -- W02: cardholder sign-off name and date swapped in one contiguous block
    swapped = w["CH Signoff Date"].map(lambda x: isinstance(x, str))
    serial = pd.to_numeric(w.loc[swapped, "CH Signoff Full Name"])
    names = w.loc[swapped, "CH Signoff Date"]
    assert (names == w.loc[swapped, "CH Full Name"]).all()
    fixed = (pd.Timestamp("1899-12-30") + pd.to_timedelta(serial, unit="D")).dt.round("s")
    w.loc[swapped, "CH Signoff Date"] = fixed.map(lambda t: t.to_pydatetime())
    w.loc[swapped, "CH Signoff Full Name"] = names
    add_flag(w, swapped, "signoff_swap_repaired")
    log("W02", "works", "CH Signoff Full Name / CH Signoff Date",
        "Columns swapped: name stored in the date column, Excel date serial stored in the name column",
        swapped.sum(), "fixed",
        f"One contiguous block (original sheet Excel rows {w.loc[swapped, 'source_row'].min()}-"
        f"{w.loc[swapped, 'source_row'].max()}, posted Jul-Sep 2026). Swapped back; serials converted to "
        f"timestamps ({fixed.min()} to {fixed.max()}), all 1-39 days after posting and before manager sign-off.")

    for c in ["Post Date", "Purchase Date", "CH Signoff Date", "Mgr Signoff Date"]:
        w[c] = pd.to_datetime(w[c])

    # -- W03: codes stored as a mix of numbers and text
    mixed = {c: w[c].map(lambda x: type(x).__name__).nunique() > 1
             for c in ["GL: Fund", "GL: Department", "GL: Unit", "GL: Object", "GL: Activity"]}
    for c in ["GL: Fund", "GL: Department", "GL: Unit", "GL: Object", "GL: Activity"]:
        w[c] = code4(w[c])
    w["GL: Task Order"] = squash(w["GL: Task Order"])
    for c in ["MCC", "Comp Supp MCC"]:
        w[c] = w[c].map(lambda x: str(int(x)).zfill(4))
    log("W03", "works", "GL: Fund, GL: Department, GL: Unit, GL: Object, GL: Activity, MCC",
        "Codes stored as numbers in some rows and text in others (changes by year and by release)",
        len(w), "fixed", "All codes stored as text. " + ", ".join(k for k, v in mixed.items() if v) + " had mixed types.")

    # -- W04: placeholder zeros
    z = w["Anon Manager"].map(lambda x: not isinstance(x, str))
    assert w.loc[z, "Mgr Signoff Date"].isna().all()
    w.loc[z, "Anon Manager"] = None
    log("W04", "works", "Anon Manager", "Numeric 0 used where there is no manager sign-off", z.sum(), "fixed",
        "Set to blank. Every such row also has a blank Mgr Signoff Date.")
    for c, iid in [("GL: Object", "W05"), ("GL: Activity", "W06")]:
        z = w[c].isin(["0", "0000"])
        w.loc[z, c] = None
        add_flag(w, z, f"{c.split(': ')[1].lower()}_code_zero_blanked")
        log(iid, "works", c, "Code 0 used as a placeholder (transaction never allocated to an account)", z.sum(),
            "fixed", "Set to blank and flagged; descriptions on these rows are also blank.")
    odd = w["GL: Activity"].notna() & ~w["GL: Activity"].fillna("").str.fullmatch(r"\d{4}")
    log("W07", "works", "GL: Activity", "Activity code is not four digits", odd.sum(), "left as released",
        "Value(s): " + ", ".join(sorted(w.loc[odd, "GL: Activity"].unique())) + ". The intended code cannot be determined.")
    z = w["GL: Department"].isna()
    w.loc[z, "GL: Department"] = "1100"
    add_flag(w, z, "department_filled")
    log("W08", "works", "GL: Department", "Blank department", z.sum(), "fixed",
        "Filled with 1100: every other row in all three sheets is department 1100 (Austin Energy).")

    # -- W09: whitespace
    text_cols = ["Vendor Name", "Item Description", "GL: Unit Desc", "GL: Object Desc", "GL: Activity Desc",
                 "CH Full Name", "CH Signoff Full Name", "MCC Description", "Comp Supp City", "Comp Supp Country",
                 "Comp Supp Name", "Comp Supp State", "DBA Name"]
    changed = 0
    for c in text_cols:
        new = squash(w[c])
        changed += (new.fillna("") != w[c].fillna("").astype(str)).sum()
        w[c] = new
    log("W09", "works", ", ".join(text_cols[:5]) + ", ...",
        "Leading/trailing spaces, repeated spaces, and line breaks inside text cells", changed, "fixed",
        "Trimmed; runs of whitespace collapsed to a single space (16 descriptions contained line breaks). "
        "Merchant descriptors such as 'PAYPAL  ATXSOLUTION' become 'PAYPAL ATXSOLUTION'.")

    # -- W10: country codes
    cmap = {"US": "USA", "840": "USA", "PR": "PRI", "CA": "CAN", "NL": "NLD", "SG": "SGP", "0": "USA"}
    z = w["Comp Supp Country"].isin(cmap)
    w.loc[z, "Comp Supp Country"] = w.loc[z, "Comp Supp Country"].map(cmap)
    log("W10", "works", "Comp Supp Country",
        "Same country coded several ways (US / USA / 840; PR / PRI; CA / CAN; NL / NLD; SG / SGP; 0)", z.sum(),
        "fixed", "Standardized to ISO 3166 alpha-3. 'CA' rows have Canadian provinces; the three '0' rows are in Texas.")
    z = w["Comp Supp State"].notna() & ~w["Comp Supp State"].fillna("").str.fullmatch(r"[A-Z]{2}")
    log("W11", "works", "Comp Supp State", "Value is not a two-letter state code", z.sum(), "left as released",
        "Foreign region codes and card-network placeholders (" + ", ".join(sorted(w.loc[z, 'Comp Supp State'].unique())) + ").")
    z = w["Comp Supp City"].fillna("").str.contains(r"\d{3}-?\d{4}|\.com|www\.|@", case=False)
    log("W12", "works", "Comp Supp City", "City field holds a phone number, URL, or e-mail address instead of a city",
        z.sum(), "left as released",
        "Normal for card-not-present and payment-facilitator merchants; the value is itself informative.")

    # -- W13/W14: description lookups
    for code, desc, iid in [("GL: Unit", "GL: Unit Desc", "W13"), ("GL: Object", "GL: Object Desc", "W14"),
                            ("GL: Activity", "GL: Activity Desc", "W15")]:
        placeholder = w[desc].fillna("").str.startswith("Multiple unit descriptions")
        known = w.loc[w[desc].notna() & ~placeholder, [code, desc]]
        counts = known.groupby([code, desc]).size().rename("n").reset_index()
        counts["lower"] = counts[desc].str.lower()
        modal = counts.sort_values("n", ascending=False).drop_duplicates(code).set_index(code)[desc]
        n_lower = counts.groupby(code)["lower"].nunique()
        target = w[code].map(modal)
        # (a) placeholder or blank description where the code has one description elsewhere
        unambiguous = w[code].map(n_lower).fillna(0).eq(1)
        fill = (w[desc].isna() | placeholder) & target.notna() & unambiguous
        # (b) same description, different capitalisation
        case_only = (w[desc].notna() & ~placeholder & target.notna() & (w[desc].fillna("") != target.fillna(""))
                     & (w[desc].fillna("").str.lower() == target.fillna("").str.lower()))
        n_placeholder = int((placeholder & fill).sum())
        w.loc[fill | case_only, desc] = target[fill | case_only]
        add_flag(w, fill, f"{desc.split(': ')[1].lower().replace(' ', '_')}_filled")
        still_blank = w[desc].isna() & w[code].notna()
        still_placeholder = w[desc].fillna("").str.startswith("Multiple unit descriptions")
        conflict = w[desc].notna() & ~still_placeholder & (w[desc].fillna("") != w[code].map(modal).fillna(""))
        log(iid, "works", desc,
            "Description blank, replaced by a lookup placeholder, or capitalised inconsistently", (fill | case_only).sum(),
            "fixed",
            f"{int(fill.sum()) - n_placeholder:,} blank and {n_placeholder:,} placeholder ('Multiple unit descriptions from various "
            f"departments') descriptions filled from the same code's description on other rows, only where that code has "
            f"exactly one description; {int(case_only.sum()):,} capitalisation variants set to the most common spelling.")
        log(iid + "b", "works", desc, "Description still blank, placeholder, or different from the code's usual description",
            (still_blank | still_placeholder | conflict).sum(), "left as released",
            f"{int(still_blank.sum()):,} blank with no usable lookup, {int(still_placeholder.sum()):,} placeholder, "
            f"{int(conflict.sum()):,} where the same code carries a genuinely different description "
            "(the chart of accounts changed over time, or code and description were keyed separately).")

    # -- W16: cardholder name variants
    w["cardholder_name_as_released"] = w["CH Full Name"]
    name_map = standardize_works_names(w["CH Full Name"])
    w["CH Full Name"] = w["CH Full Name"].map(name_map)
    z = w["CH Full Name"] != w["cardholder_name_as_released"]
    add_flag(w, z, "cardholder_name_standardized")
    merged = {k: v for k, v in name_map.items() if k != v}
    log("W16", "works", "CH Full Name",
        "Same cardholder under two spellings (middle name dropped, mostly in rows posted July-September 2026)",
        z.sum(), "fixed",
        f"{len(merged)} short names mapped to the longer spelling when the given names are a prefix of exactly one "
        "longer name with the same surname (e.g. 'Gray, Julia' -> 'Gray, Julia M'). Released spelling kept in "
        "cardholder_name_as_released. Names that differ in a middle initial are left separate.")
    w["CH Signoff Full Name"] = w["CH Signoff Full Name"].map(lambda x: name_map.get(x, x))

    # -- W17: amount is the transaction total repeated on every line
    w["Allocated Amount"] = pd.to_numeric(w["Allocated Amount"]).round(2)
    g = w.groupby("Txn ID", sort=False)
    w["txn_line_number"] = g.cumcount() + 1
    w["txn_line_count"] = g["Txn ID"].transform("size")
    assert (g["Allocated Amount"].nunique() == 1).all()
    multi = w["txn_line_count"] > 1
    w["is_primary_line"] = w["txn_line_number"] == 1
    overstated = w.loc[~w["is_primary_line"], "Allocated Amount"].sum()
    log("W17", "works", "Allocated Amount",
        "Not an allocation: the full transaction amount is repeated on every line of a multi-line transaction",
        multi.sum(), "fixed (helper columns)",
        f"All {w.loc[multi, 'Txn ID'].nunique():,} multi-line transactions carry one identical amount on every line "
        f"(e.g. a $103.18 grocery purchase listed on 47 item lines). Summing the column overstates spending by "
        f"${overstated:,.2f}. Added txn_line_number, txn_line_count and is_primary_line; total spending = "
        "SUM(amount) WHERE is_primary_line. Amounts themselves are unchanged.")

    # -- W18: exact duplicate lines
    log("W18", "works", "(all fields)", "Line is identical in every released field to an earlier line of the same transaction",
        w["is_exact_duplicate_line"].sum(), "flagged",
        f"{n_dup_txns} transactions. Kept, and flagged in "
        "is_exact_duplicate_line. Because the amount is a repeated transaction total (W17), these lines do not inflate "
        "spending once is_primary_line is used.")

    # -- W19: split transactions
    w["txn_id_base"] = w["Txn ID"].str.replace(r"-[A-Z]$", "", regex=True)
    z = w["Txn ID"] != w["txn_id_base"]
    log("W19", "works", "Txn ID", "Transaction ID carries a -A/-B/-C suffix", z.sum(), "documented",
        f"{w.loc[z, 'txn_id_base'].nunique()} card charges were split in Works into separately coded parts; here the "
        "parts DO carry different amounts that add up to the charge. txn_id_base holds the ID without the suffix.")

    # -- sign-off gaps and timing (documented, not fixable)
    t = w[w["is_primary_line"]]
    log("W20", "works", "CH Signoff Date", "No cardholder sign-off recorded", t["CH Signoff Date"].isna().sum(),
        "left blank", "Counted in transactions. "
        f"{int((t['CH Signoff Date'].isna() & (t['Post Date'] >= '2026-07-01')).sum()):,} were posted July-September 2026 and "
        "were probably still open when the data was extracted.")
    log("W21", "works", "Mgr Signoff Date", "No manager sign-off recorded", t["Mgr Signoff Date"].isna().sum(),
        "left blank", "Counted in transactions. "
        f"{int((t['Mgr Signoff Date'].isna() & (t['Post Date'] >= '2026-07-01')).sum()):,} were posted July-September 2026. "
        "Absence in the extract does not prove the purchase was never reviewed.")
    lag = (t["Post Date"] - t["Purchase Date"]).dt.days
    log("W22", "works", "Purchase Date / Post Date", "Posted more than 30 days after purchase", (lag > 30).sum(),
        "left as released", "Mostly dispute and claim adjustments. No transaction posts before its purchase date, and no "
        "sign-off precedes posting.")
    log("W23", "works", "Comp Supp MCC", "Redundant column", len(w), "kept",
        "Identical to MCC on every row. Kept under the name supplier_mcc for fidelity to the release.")
    log("W24", "works", "Item Description", "Characters outside plain ASCII (accents, curly quotes, dashes)",
        w["Item Description"].fillna("").str.contains(r"[^\x00-\x7f]").sum(), "left as released",
        "Valid Unicode; files are written as UTF-8.")
    log("W25", "works", "MCC Description", "Truncated at 40 characters, with source typos (e.g. 'NOT ESLEWHERE CLASSI')",
        (w["MCC Description"].str.len() == 40).sum(), "left as released", "Card-network text as released.")
    return w, name_map


def standardize_works_names(names):
    """Map 'Last, First' to 'Last, First Middle' when it matches exactly one longer name."""
    uniq = sorted(names.dropna().unique())
    parsed = {}
    for n in uniq:
        last, given = [p.strip() for p in n.split(",", 1)]
        parsed[n] = (last.lower(), [g.lower().rstrip(".") for g in given.split()])
    out = {n: n for n in uniq}

    def is_prefix(short, long):
        if len(short) >= len(long):
            return False
        return all(a == b or (len(a) == 1 and b.startswith(a)) for a, b in zip(short, long))

    for n in uniq:
        last, given = parsed[n]
        longer = [m for m in uniq if m != n and parsed[m][0] == last and is_prefix(given, parsed[m][1])]
        if len(longer) == 1:
            out[n] = longer[0]
    for n in uniq:  # resolve chains
        while out[out[n]] != out[n]:
            out[n] = out[out[n]]
    return out


# --------------------------------------------------------------------------- Legacy
def load_legacy(original, works_names):
    m = pd.read_excel(original, sheet_name=LEGACY_SHEET, dtype=object)
    m["source_file"], m["source_sheet"] = ORIGINAL_NAME, LEGACY_SHEET
    m["source_row"] = np.arange(len(m)) + 2
    m["dq_flags"] = ""
    desc = m["Accounting Line Desc"].astype(str)
    assert desc.str.len().min() > 30 and (desc.str[29] == " ").mean() > 0.99
    m["vendor_name"] = squash(desc.str[:30])
    m["item_description"] = squash(desc.str[30:])
    log("L01", "legacy", "Accounting Line Desc",
        "Merchant name and purchase description packed into one fixed-width field", len(m), "fixed",
        "Characters 1-30 -> vendor_name, 31-100 -> item_description (whitespace trimmed).")
    log("L02", "legacy", "Accounting Line Desc", "Purchase description cut off at 70 characters",
        (desc.str.len() == 100).sum(), "left as released", "The field is capped at 100 characters in the source system.")
    for c in ["Fund", "Unit", "Object Code", "Activity", "Accounting Line"]:
        m[c] = m[c].astype(str).str.strip()
    m["Department Code"] = m["Department Code"].map(lambda x: str(int(x)))
    m["Taskorder"] = squash(m["Taskorder"])
    for c in ["Unit Name", "Object Name", "Activity Name"]:
        m[c] = squash(m[c])
    m["Posting Amount"] = pd.to_numeric(m["Posting Amount"]).round(2)

    # names: 'MARK ANTHONY YBARRA' -> 'Ybarra, Mark Anthony', reusing the Works spelling where the person appears there
    def works_key(n):
        last, given = [p.strip() for p in n.split(",", 1)]
        return f"{given} {last}".upper()

    exact = {works_key(n): n for n in works_names}
    first_last = {}
    for n in works_names:
        last, given = [p.strip() for p in n.split(",", 1)]
        first_last.setdefault((given.split()[0].upper(), last.upper()), []).append(n)

    def convert(raw):
        key = re.sub(r"\s+", " ", raw.strip().upper())
        if key in exact:
            return exact[key], "exact"
        toks = key.split()
        cands = first_last.get((toks[0], toks[-1]), [])
        if len(cands) == 1:
            return cands[0], "first_last"
        return f"{toks[-1].title()}, {' '.join(t.title() for t in toks[:-1])}", "reformatted"

    conv = {n: convert(n) for n in m["Document Description"].unique()}
    m["cardholder_name_as_released"] = m["Document Description"]
    m["cardholder_name"] = m["Document Description"].map(lambda n: conv[n][0])
    how = pd.Series({k: v[1] for k, v in conv.items()}).value_counts().to_dict()
    log("L03", "legacy", "Document Description",
        "Cardholder name written 'FIRST MIDDLE LAST' in capitals; newer system uses 'Last, First Middle'", len(m),
        "fixed",
        f"{len(conv)} names converted to the newer format so a cardholder can be followed across both systems: "
        f"{how.get('exact', 0)} matched a newer-system name exactly, {how.get('first_last', 0)} matched on first and last "
        f"name (unique match), {how.get('reformatted', 0)} do not appear in the newer system and were reformatted "
        "assuming the final word is the surname. Released spelling kept in cardholder_name_as_released.")
    log("L04", "legacy", "Fiscal Month",
        "No calendar dates; fiscal month 1 never appears and month 12 holds about twice the usual volume",
        len(m), "documented",
        "Each row is a line of a monthly ProCard journal voucher, dated only by accounting period. Dates quoted inside "
        "descriptions fall mostly in the calendar month before the period, so period N holds roughly the statement "
        "for month N-1 and period 12 holds two statements (year-end accrual). No dates were invented.")
    log("L05", "legacy / works", "coverage",
        "Probable gap of about one statement cycle between the two systems (December 2017)", 0, "not fixable",
        "Legacy rows end at fiscal year 2018 period 3 (December 2017 accounting period, purchases to about late "
        "November 2017). Newer-system rows start with post date 2018-01-02 (purchase dates from 2017-12-27). "
        "Purchases posted in December 2017 appear in neither sheet; a follow-up request would be needed.")
    log("L06", "legacy", "(grain)", "Different grain and fields from the newer system", len(m), "documented",
        "Legacy rows have no transaction ID, dates, merchant category, or sign-off data. Amounts are per accounting "
        "line and can be summed. Use source_system to keep the two populations apart.")
    return m


# --------------------------------------------------------------------------- Assemble
def fiscal(d):
    fy = d.dt.year + (d.dt.month >= 10).astype(int)
    fm = (d.dt.month - 10) % 12 + 1
    return fy, fm


COLUMNS = [
    "record_id", "source_system", "source_file", "source_sheet", "source_row",
    "cardholder_name", "cardholder_name_as_released",
    "fiscal_year", "fiscal_month", "purchase_date", "post_date",
    "txn_id", "txn_id_base", "txn_line_number", "txn_line_count", "is_primary_line", "is_exact_duplicate_line",
    "document_number", "accounting_line",
    "vendor_name", "item_description", "amount",
    "gl_fund", "gl_department", "gl_unit", "gl_unit_desc", "gl_object", "gl_object_desc",
    "gl_task_order", "gl_activity", "gl_activity_desc",
    "ch_signoff_name", "ch_signoff_date", "mgr_signoff_date", "anon_manager",
    "mcc", "mcc_description", "supplier_name", "supplier_city", "supplier_state", "supplier_country",
    "supplier_mcc", "dba_name", "dq_flags",
]


def assemble(w, m):
    fy, fm = fiscal(w["Post Date"])
    W = pd.DataFrame({
        "source_system": "works", "source_file": w["source_file"], "source_sheet": w["source_sheet"],
        "source_row": w["source_row"],
        "cardholder_name": w["CH Full Name"], "cardholder_name_as_released": w["cardholder_name_as_released"],
        "fiscal_year": fy, "fiscal_month": fm,
        "purchase_date": w["Purchase Date"].dt.strftime("%Y-%m-%d"), "post_date": w["Post Date"].dt.strftime("%Y-%m-%d"),
        "txn_id": w["Txn ID"], "txn_id_base": w["txn_id_base"], "txn_line_number": w["txn_line_number"],
        "txn_line_count": w["txn_line_count"], "is_primary_line": w["is_primary_line"],
        "is_exact_duplicate_line": w["is_exact_duplicate_line"],
        "vendor_name": w["Vendor Name"], "item_description": w["Item Description"], "amount": w["Allocated Amount"],
        "gl_fund": w["GL: Fund"], "gl_department": w["GL: Department"], "gl_unit": w["GL: Unit"],
        "gl_unit_desc": w["GL: Unit Desc"], "gl_object": w["GL: Object"], "gl_object_desc": w["GL: Object Desc"],
        "gl_task_order": w["GL: Task Order"], "gl_activity": w["GL: Activity"], "gl_activity_desc": w["GL: Activity Desc"],
        "ch_signoff_name": w["CH Signoff Full Name"],
        "ch_signoff_date": w["CH Signoff Date"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "mgr_signoff_date": w["Mgr Signoff Date"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "anon_manager": w["Anon Manager"],
        "mcc": w["MCC"], "mcc_description": w["MCC Description"], "supplier_name": w["Comp Supp Name"],
        "supplier_city": w["Comp Supp City"], "supplier_state": w["Comp Supp State"],
        "supplier_country": w["Comp Supp Country"], "supplier_mcc": w["Comp Supp MCC"], "dba_name": w["DBA Name"],
        "dq_flags": w["dq_flags"],
    })
    W["_order"] = np.arange(len(W))
    W = W.sort_values(["post_date", "txn_id", "_order"], kind="stable").drop(columns="_order")
    L = pd.DataFrame({
        "source_system": "legacy", "source_file": m["source_file"], "source_sheet": m["source_sheet"],
        "source_row": m["source_row"],
        "cardholder_name": m["cardholder_name"], "cardholder_name_as_released": m["cardholder_name_as_released"],
        "fiscal_year": m["Fiscal Year"].astype(int), "fiscal_month": m["Fiscal Month"].astype(int),
        "is_primary_line": True, "is_exact_duplicate_line": False,
        "document_number": m["Document Number"], "accounting_line": m["Accounting Line"],
        "vendor_name": m["vendor_name"], "item_description": m["item_description"], "amount": m["Posting Amount"],
        "gl_fund": m["Fund"], "gl_department": m["Department Code"], "gl_unit": m["Unit"],
        "gl_unit_desc": m["Unit Name"], "gl_object": m["Object Code"], "gl_object_desc": m["Object Name"],
        "gl_task_order": m["Taskorder"], "gl_activity": m["Activity"], "gl_activity_desc": m["Activity Name"],
        "dq_flags": m["dq_flags"],
    })
    out = pd.concat([L, W], ignore_index=True)
    out.insert(0, "record_id", np.arange(1, len(out) + 1))
    for c in ["txn_line_number", "txn_line_count", "fiscal_year", "fiscal_month", "source_row"]:
        out[c] = out[c].astype("Int64")
    return out[COLUMNS]


def transactions(combined):
    """One row per card charge (Works only). Split parts (-A/-B/-C) are added back together."""
    w = combined[combined["source_system"] == "works"]
    primary = w[w["is_primary_line"]]
    g = w.groupby("txn_id_base", sort=False)
    agg = pd.DataFrame({
        "amount": primary.groupby("txn_id_base", sort=False)["amount"].sum().round(2),
        "line_count": g.size(),
        "split_parts": g["txn_id"].nunique(),
        "gl_units": g["gl_unit"].agg(lambda s: "|".join(pd.unique(s.dropna()))),
        "gl_objects": g["gl_object"].agg(lambda s: "|".join(pd.unique(s.dropna()))),
        "all_item_descriptions": g["item_description"].agg(lambda s: " | ".join(pd.unique(s.dropna()))),
    })
    first = primary.drop_duplicates("txn_id_base").set_index("txn_id_base")
    keep = ["cardholder_name", "fiscal_year", "fiscal_month", "purchase_date", "post_date", "vendor_name",
            "gl_unit", "gl_unit_desc", "gl_object", "gl_object_desc", "gl_task_order", "gl_activity", "gl_activity_desc",
            "ch_signoff_name", "ch_signoff_date", "mgr_signoff_date", "anon_manager", "mcc", "mcc_description",
            "supplier_name", "supplier_city", "supplier_state", "supplier_country", "dba_name"]
    t = first[keep].join(agg).reset_index().rename(columns={"txn_id_base": "txn_id"})
    order = ["txn_id", "cardholder_name", "fiscal_year", "fiscal_month", "purchase_date", "post_date", "vendor_name",
             "amount", "line_count", "split_parts", "all_item_descriptions", "gl_unit", "gl_unit_desc", "gl_units",
             "gl_object", "gl_object_desc", "gl_objects", "gl_task_order", "gl_activity", "gl_activity_desc",
             "ch_signoff_name", "ch_signoff_date", "mgr_signoff_date", "anon_manager", "mcc", "mcc_description",
             "supplier_name", "supplier_city", "supplier_state", "supplier_country", "dba_name"]
    return t[order]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True)
    ap.add_argument("--supplement", required=True)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    w = load_works(a.original, a.supplement)
    n_released = len(w)
    w, name_map = clean_works(w)
    m = load_legacy(a.original, sorted(set(name_map.values())))
    combined = assemble(w, m)
    assert len(combined) == n_released + len(m)
    txns = transactions(combined)

    # reconciliation: nothing dropped, no amount altered
    raw_w = pd.concat([pd.read_excel(a.original, sheet_name=WORKS_SHEET, usecols=["Allocated Amount"]),
                       pd.read_excel(a.supplement, sheet_name=SUPP_SHEET, usecols=["Allocated Amount"])])
    cw = combined[combined.source_system == "works"]
    assert abs(raw_w["Allocated Amount"].sum() - cw["amount"].sum()) < 0.005
    assert abs(cw.loc[cw.is_primary_line, "amount"].sum() - txns["amount"].sum()) < 0.005

    combined.to_csv(out / "ae_pcard_combined.csv", index=False, encoding="utf-8", lineterminator="\n", float_format="%.2f")
    txns.to_csv(out / "ae_pcard_transactions.csv", index=False, encoding="utf-8", lineterminator="\n", float_format="%.2f")
    pd.DataFrame(ISSUES).to_csv(out / "data_quality_issues.csv", index=False, encoding="utf-8", lineterminator="\n")
    nm = pd.concat([
        w[["cardholder_name_as_released", "CH Full Name"]].set_axis(["name_as_released", "cardholder_name"], axis=1).assign(source_system="works"),
        m[["cardholder_name_as_released", "cardholder_name"]].set_axis(["name_as_released", "cardholder_name"], axis=1).assign(source_system="legacy"),
    ]).drop_duplicates().sort_values(["cardholder_name", "source_system"])
    nm.to_csv(out / "cardholder_name_map.csv", index=False, encoding="utf-8", lineterminator="\n")

    print(f"combined rows: {len(combined):,}  (legacy {len(m):,} + works {n_released:,})")
    print(f"works transactions (card charges): {len(txns):,}; net amount ${txns['amount'].sum():,.2f}")
    print(f"sum of released Allocated Amount column: ${cw['amount'].sum():,.2f}")
    print(f"legacy net amount: ${combined.loc[combined.source_system == 'legacy', 'amount'].sum():,.2f}")
    print(f"cardholders: {combined['cardholder_name'].nunique()}  issues logged: {len(ISSUES)}")


if __name__ == "__main__":
    main()
