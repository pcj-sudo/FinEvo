# Aster Vale Products – Forecast and Debt-Sweep Modeling

Aster Vale Products is a fictional industrial company with three years of audited historical results (2020–2022) and a five-year forecast horizon (2023–2027).

## Objective

Complete the forecast for the **Calcs** schedule so that all downstream financial statements and the DCF valuation propagate correctly through 2027.

The `Key Assumptions` sheet provides the modeling assumptions and business rules. Historical actuals for 2020–2022 are locked. The forecast-period columns (F–J) in the `Calcs` sheet contain the cells you need to model.

## Scope

Your modelled cells must cover at minimum:

- **PP&E depreciation schedule** – straight-line on existing net PP&E at the end-year remaining life, plus full-year depreciation on each year's new CapEx, tracked as a triangular cohort matrix.
- **Intangible amortization** – same cohort-method logic applied to intangible assets.
- **Debt schedule and sweep** – annual LT-debt repayment up to the opening balance using available cash generation; the circular relationship between interest expense, interest income, and cash/debt balances must be resolved algebraically.
- **Interest calculations** – debt expense and cash income accrue on arithmetic-average opening/closing balances at the last historical implied annual rate.
- **Retained earnings** – net income less dividends (zero in forecast), carried forward from 2022.
- **Cash-generation and debt-sweep helpers** (rows 71–74) that feed the circular debt/interest formulas and the cash-flow statement.

The completed workbook should be saved as `/app/output.xlsx`, preserving the original sheet structure and naming conventions.
