Complete the three-statement financial forecast for Aster Vale Systems for FY2031 Q1 through FY2033 Q4 using the operating assumptions in the **Operating Drivers** sheet.

The workbook contains an integrated financial model with an Income Statement, Balance Sheet, Cash Flow statement, and a Segments revenue/gross profit breakdown. All forecast columns (1FQ2031E through 4FQ2033E) have been cleared and must be populated using formulas that derive from the operating assumptions.

Key relationships to implement:
- **Segment revenue**: Each segment's quarterly revenue is driven by sequential growth rates in *Operating Drivers*, applied to the prior quarter. FY2032E references FY2031E Q4 as the base; FY2033E references FY2032E Q4. Segment revenue rolls up to the Income Statement.
- **Gross profit**: Derived from non-GAAP gross margins, adjusted for stock-based compensation, amortization, and exceptional items per the assumptions.
- **Operating expenses**: R&D and SG&A scale with revenue per their respective ratios, with SBC and exceptional item adjustments.
- **Income tax**: Computed using the GAAP effective tax rate; adjusted and non-GAAP variants use their own rates.
- **Balance sheet**: Working-capital items use receivable/inventory/payable days from *Operating Drivers*. PP&E and intangibles roll forward with capex, depreciation, and amortization. Debt changes per issuance/repayment drivers. Share count evolves per issuance/repurchase/vesting mechanics.
- **Cash flow**: Operating CF starts from net income and adjusts for noncash items and working-capital changes. Capex and financing activities complete the statement; ending cash must tie to the Balance Sheet.
- **Non-GAAP measures**: Non-GAAP revenue, income, EPS, and EBITDA follow the reconciliation structure.
- **Annual summaries**: FY2031E, FY2032E, and FY2033E annual columns aggregate the quarterly forecasts.

All operating assumptions are in **Operating Drivers**. Business conventions are in **Business Definitions**. Preserve workbook structure, number formats, and formula conventions.

Save the completed workbook as `/app/output.xlsx`.
