# Aster Vale Systems — 3-Year Financial Forecast Completion

**Company:** Aster Vale Systems (fictional semiconductor & infrastructure software company)
**Fiscal year-end:** September 30
**Currency / shares:** millions (USD-equivalent)

## Task Description

You are given a quarterly financial model for Aster Vale Systems spanning FY2011 through FY2033.
The model includes an Income Statement, Balance Sheet, Cash Flow Statement, Segments breakdown,
and an Operating Drivers sheet containing all input assumptions (revenue growth rates, margins,
operating ratios, tax rates, balance sheet metrics, and capital structure parameters).

The **actual history** covers FY2011A through FY2030A (80 quarters).
The **forecast period** is FY2031E through FY2033E (12 quarters).

All forecast-period cells have been blanked across the Income Statement, Balance Sheet,
Cash Flow, and Segments sheets, including annual aggregates (FY2031E–FY2033E) and
calendar year aggregates (CY2030E–CY2032E). The Operating Drivers sheet retains all
input assumptions (growth rates, margins, ratios, share counts, tax rates, etc.)
for the forecast period.

## What to Do

1. Study the **Operating Drivers** sheet to understand all input assumptions.
2. Study the **historical data** (FY2011A–FY2030A) across all sheets to understand
   the calculation methodology, formula patterns, and cross-sheet dependencies.
3. Rebuild the **complete forecast model** for all 12 quarters (1FQ2031E through
   4FQ2033E), plus the annual and calendar year aggregates, across:
   - **Segments**: segment-level revenue and gross profit from growth rates and margins
   - **Income Statement**: revenue, COGS, operating expenses, interest, tax, EPS
   - **Balance Sheet**: rollforward of assets, liabilities, equity, and ratios
   - **Cash Flow**: reconciliation from net income to free cash flow

Key computation areas include:
- **Revenue build-up** from segment sequential growth rates (Operating Drivers)
- **Gross margin & COGS** with amortization allocation
- **Operating expenses** (R&D, SG&A) from revenue ratios plus stock compensation
- **Interest income/expense** from cash and debt balances
- **Tax computation** using effective rates and tax bridges
- **EPS** from share counts, dilution, and net income
- **Balance sheet rollforwards**: receivables, inventory, payables, PP&E, debt, equity
- **Cash flow** reconciliation from income statement + balance sheet changes
- **Cross-sheet dependencies**: Segments → Income Statement → Balance Sheet → Cash Flow

## Deliverable

Save the completed workbook as `/app/output.xlsx`.

Preserve the existing structure, conventions, and historical data. Only fill in the blank
forecast cells. All assumptions from the Operating Drivers sheet must be applied consistently.
