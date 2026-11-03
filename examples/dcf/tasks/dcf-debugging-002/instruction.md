# Aster Vale Products — DCF Valuation Model Audit

You are given a multi-sheet financial model for a fictional manufacturing company (Aster Vale Products). The model projects income statement, balance sheet, cash flow statement, and a discounted cash flow (DCF) valuation through 2027, using 2020–2022 historicals and 2023–2027 forecasts.

The model contains errors in its calculations. Your task is to identify and correct all errors so that the model's calculations are consistent with the supplied business assumptions and the model's own internal logic.

## What to preserve

- All given assumptions, labels, row structure, and sheet names must remain unchanged.
- Historical actuals (2020–2022) are fixed inputs — do not modify them except where a formula overwrites a cell that should contain a hard-coded historical value.
- The model's business rules (described in *Key Assumptions* sheet) are correct; errors are only in formulas that implement those rules.

## What to fix

Errors may include incorrect cell references, missing components in formulas, or wrong parameters. Some errors affect only a single cell; others propagate through multiple sheets. The DCF valuation sheet depends on the income statement, balance sheet, and calculation schedules.

## Deliverable

Save the corrected workbook as `/app/output.xlsx`.
