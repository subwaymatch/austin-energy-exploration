# Austin Energy ProCard data, combined and cleaned

Purchasing-card (ProCard) records for Austin Energy cardholders, obtained from the City of Austin under the
Texas Public Information Act and combined into one file.

| Release | File as released | Sheet | Rows | Covers |
|---|---|---|---|---|
| C340726-102326 (received 2026-11-06) | `C340726_AE.xlsx` | `PIR - MS Data 2013-2018` | 30,744 | Older system, fiscal years 2013–2018 (through accounting period 3 of FY2018) |
| C340726-102326 | `C340726_AE.xlsx` | `PIR - Works Data 2018-2026` | 112,641 | Newer system, posted 2018-01-02 to 2026-09-30, except January–June 2019 |
| C343310-110726 (received 2026-11-13) | `C343310.xlsx` | `Sheet1` | 7,959 | Newer system, posted January–June 2019 |

The City redacted confidential and competitive information before release. The original workbooks are not in this
repository; `scripts/build_combined.py` rebuilds everything here from them.

## Files

| File | Rows | What it is |
|---|---|---|
| `ae_pcard_combined.csv` | 151,344 | **The combined file.** Every released row from all three sheets in one set of columns. One row = one accounting line. |
| `ae_pcard_transactions.csv` | 116,842 | Derived from the combined file: one row per card charge, newer system only. Use this for anything that counts or sums transactions. |
| `data_quality_issues.csv` | 34 | Every issue found, how many rows it touches, and what was done. |
| `cardholder_name_map.csv` | 792 | Released spelling of each cardholder name and the standardized name used here. |

All files are UTF-8 CSV with a header row. Blank means not available. Read code columns (`gl_*`, `mcc`) as text.

## Read this before adding up amounts

In the newer system the released column `Allocated Amount` is **not** an allocation. When a transaction has several
lines, every line repeats the full transaction amount: all 1,716 multi-line transactions carry one identical amount on
every line. Summing the column gives $65,946,679.44; the true net total is $63,908,807.05.

- In `ae_pcard_combined.csv`, sum `amount` only where `is_primary_line` is `True`.
- Or use `ae_pcard_transactions.csv`, where each charge appears once.

Older-system rows are individual accounting lines and can be summed directly (`is_primary_line` is always `True`).

The two systems are different populations. The older one has no transaction IDs, dates, merchant categories, or
sign-off data, and its fiscal periods lag the purchase month. Filter on `source_system` rather than pooling them.

## Columns in `ae_pcard_combined.csv`

| Column | Older system | Newer system |
|---|---|---|
| `record_id` | Row number in this file (1–151,344) | same |
| `source_system` | `legacy` | `works` |
| `source_file`, `source_sheet`, `source_row` | Where the row came from; `source_row` is the Excel row number | same |
| `cardholder_name` | Standardized to `Last, First Middle` | Standardized (middle-name variants merged) |
| `cardholder_name_as_released` | `Document Description` | `CH Full Name` |
| `fiscal_year`, `fiscal_month` | As released: the accounting period of the monthly journal voucher | Derived from `post_date` (fiscal year starts October 1; month 1 = October) |
| `purchase_date`, `post_date` | blank | `Purchase Date`, `Post Date` (YYYY-MM-DD) |
| `txn_id` | blank | `Txn ID` |
| `txn_id_base` | blank | `Txn ID` without a `-A`/`-B`/`-C` split suffix |
| `txn_line_number`, `txn_line_count` | blank | Position of the line within its transaction, and number of lines |
| `is_primary_line` | always `True` | `True` on the first line of each transaction |
| `is_exact_duplicate_line` | always `False` | `True` if the line repeats an earlier line of the same transaction in every released field |
| `document_number`, `accounting_line` | `Document Number`, `Accounting Line` | blank |
| `vendor_name` | First 30 characters of `Accounting Line Desc` | `Vendor Name` |
| `item_description` | Characters 31–100 of `Accounting Line Desc` (cut off at 70 characters in the source) | `Item Description` |
| `amount` | `Posting Amount` | `Allocated Amount` (see warning above) |
| `gl_fund`, `gl_department` | `Fund`, `Department Code` | `GL: Fund`, `GL: Department` |
| `gl_unit`, `gl_unit_desc` | `Unit`, `Unit Name` | `GL: Unit`, `GL: Unit Desc` |
| `gl_object`, `gl_object_desc` | `Object Code`, `Object Name` | `GL: Object`, `GL: Object Desc` |
| `gl_task_order` | `Taskorder` | `GL: Task Order` |
| `gl_activity`, `gl_activity_desc` | `Activity`, `Activity Name` | `GL: Activity`, `GL: Activity Desc` |
| `ch_signoff_name`, `ch_signoff_date` | blank | `CH Signoff Full Name`, `CH Signoff Date` (YYYY-MM-DD HH:MM:SS) |
| `mgr_signoff_date`, `anon_manager` | blank | `Mgr Signoff Date`, `Anon Manager` (ID assigned by the City) |
| `mcc`, `mcc_description` | blank | `MCC`, `MCC Description` (merchant category code) |
| `supplier_name`, `supplier_city`, `supplier_state`, `supplier_country`, `supplier_mcc` | blank | `Comp Supp Name`, `... City`, `... State`, `... Country`, `... MCC` |
| `dba_name` | blank | `DBA Name` |
| `dq_flags` | blank | Repairs applied to this row, separated by `;` |

`supplier_city` often holds a phone number or web address instead of a city. That is how card networks report
online and payment-processor merchants, and it is left as released.

## What was repaired

Full detail is in `data_quality_issues.csv`. No row was dropped and no amount was changed.

- **Missing months.** The first release had no rows posted January–June 2019; the second release fills them, with no overlapping transaction IDs.
- **Swapped sign-off columns.** In 1,237 consecutive rows (posted July–September 2026) the cardholder sign-off name and date were in each other's columns. Swapped back.
- **Cardholder names.** 47 people appear under two spellings (middle name dropped, mostly in the last three months). Merged; older-system names converted from `FIRST MIDDLE LAST`.
- **Codes.** Accounting codes and MCCs were stored as numbers in some years and text in others. All text now. Placeholder zeros blanked.
- **Country codes.** `US` / `USA` / `840` and similar variants standardized to three-letter codes.
- **Descriptions.** Blank or placeholder unit and activity descriptions filled from the same code on other rows, only where that code has exactly one description.
- **Whitespace.** Trimmed; repeated spaces and line breaks inside cells collapsed.
- **Older-system description field.** Split into merchant name and purchase description.

## Known gaps that could not be fixed

- **December 2017.** Older-system rows end with the December 2017 accounting period; newer-system rows start with charges posted 2018-01-02. About one statement cycle is in neither.
- **No dates in the older system.** Fiscal period 1 never appears and period 12 holds about two months of activity, which points to a one-month posting lag with a year-end accrual. No dates were invented.
- **No fraud labels.** Nothing in either release marks which transactions the City Auditor counted as fraudulent.
- **Sign-offs missing for recent months.** About 1,490 transactions posted July–September 2026 had no sign-off when the data was extracted.

## Rebuild

```bash
python scripts/build_combined.py --original /path/to/C340726_AE.xlsx --supplement /path/to/C343310.xlsx --out data
```
