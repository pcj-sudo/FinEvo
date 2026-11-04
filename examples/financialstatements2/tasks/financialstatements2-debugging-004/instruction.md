# Quarterly Financial Model — Audit and Correction

You are provided with a quarterly financial model covering the period from Q1 FY2001 through Q3 FY2003. The workbook contains an income statement (`IS`), a balance sheet (`BS`), a cash-flow statement (`CF`), driver assumptions (`Drivers`), debt facility details (`Debt Terms`), and opening balance information (`Opening`).

## Objective

Audit the model and correct **all calculation errors** that affect the financial statements. The supplied assumptions (driver values, debt terms, opening balances) are correct and must be preserved. The workbook structure (sheet names, row labels, column layout) should remain unchanged.

After correcting every error, save the repaired workbook as:

```
/app/output.xlsx
```

## Scoring

Your submission will be evaluated by comparing specific cells in your corrected workbook against expected values. Only the targeted cells determine the score; differences outside those cells are diagnostic only.

The evaluation uses exact matching on the specified cells after rounding to two decimal places (ROUND_HALF_UP).
