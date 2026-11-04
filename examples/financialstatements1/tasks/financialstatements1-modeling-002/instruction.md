# Complete Financial Forecast Model

Rebuild the bottoms-up segment forecast and the consolidated income statement
forecast for Aster Vale Systems using the supplied operating assumptions.

## Background

The workbook contains a multi-period financial model for Aster Vale Systems
with the following key sheets:

- **Operating Drivers**: Sequential growth rates by product, margin assumptions
  by business segment, tax rates, expense ratios, and share count assumptions.
- **Segments**: Bottoms-up revenue and gross profit forecasts by product,
  aggregated into semi solutions and infrastructure software segments,
  with cross-checks against top-down income statement figures.
- **Income Statement**: Full GAAP and non-GAAP income statement, including
  COGS, operating expenses, interest, tax, net income, EPS, and reconciliations.
- **Balance Sheet** and **Cash Flow**: Supporting statement with rollforward
  logic and cross-referenced data.

## Your Task

The forecast columns from **1FQ2031E** through **2FQ2032E** (columns AN through
AS) have been cleared on both the **Segments** and **Income Statement** sheets.

Reconstruct all blanked forecast cells using:

- The operating assumptions provided on the **Operating Drivers** sheet.
- The historical formula patterns visible in the prior columns (AM and earlier).
- Cross-sheet references to **Balance Sheet**, **Cash Flow**, and
  **Operating Drivers** as shown in the historical data.

Specifically, complete on the **Segments** sheet:

- Product-level revenue for each of the 9 products (chained sequential growth).
- Sequential (Q-o-Q) growth rates linked from Operating Drivers.
- Gross margin rates linked from Operating Drivers.
- Bottoms-up aggregations: semi solutions, infrastructure software,
  total revenue, gross profit, and derived margin percentages.

And on the **Income Statement** sheet:

- Revenue, COGS, and gross profit (GAAP and adjusted).
- Operating expenses (R&D, SG&A, other OpEx) with one-time item adjustments.
- Operating income and margins.
- Interest income/expense (including debt-averaging formulas referencing
  the Balance Sheet).
- Tax, net income, and EPS calculations.
- Non-GAAP measures and reconciliations.

Preserve the existing structure, check cells, and cross-references. The check
cells on the Segments sheet should read "OK" when your calculations are correct.

Save the completed workbook as `/app/output.xlsx`.
