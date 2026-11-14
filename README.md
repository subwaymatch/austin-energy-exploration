# austin-energy-exploration

Exploration of Austin Energy purchasing-card (ProCard) transactions, obtained from the City of Austin through two
public information requests, as groundwork for a business-analytics and accounting-analytics case study.

The reference case is the Austin City Auditor's investigation IN24001, *Austin Energy Fraud and Waste* (December 2025),
which found that a former Facility Service Specialist reported about $980,000 of card transactions to fictitious vendors
over roughly six years.

## What is here

| Path | Contents |
|---|---|
| `data/ae_pcard_combined.csv` | Both releases combined and cleaned: 151,344 accounting lines, fiscal years 2013–2026 |
| `data/ae_pcard_transactions.csv` | One row per card charge for the newer system (2018–2026) |
| `data/data_quality_issues.csv` | Every data issue found and what was done about it |
| `data/README.md` | Column dictionary, provenance, and the one thing to know before summing amounts |
| `reports/austin_energy_pcard_fraud_detection.html` | Findings: Benford's Law and ten other tests, and how early each would have flagged the card |
| `analysis/` | Scores per cardholder, the subject's transactions with flags, and all report numbers as JSON |
| `scripts/` | `build_combined.py` → `analyze.py` → `build_report.py` |

## Reproduce

```bash
pip install pandas openpyxl numpy scipy scikit-learn
```

```bash
python scripts/build_combined.py --original /path/to/C340726_AE.xlsx --supplement /path/to/C343310.xlsx --out data
```

```bash
python scripts/analyze.py
```

```bash
python scripts/build_report.py
```

Cardholders other than the subject of the auditor's report, and the payees on his card, appear under aliases in
`analysis/` and in the report. A high score is a reason to look at receipts, not a finding about a person.
`python scripts/analyze.py --show-names` uses the names as the City released them.
