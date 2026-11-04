# Aster Vale Systems — Financial Forecast Model

You are given a financial model workbook for **Aster Vale Systems**, a semiconductor and infrastructure-software company with a September fiscal year-end. The workbook includes historical data and operating-driver assumptions that support a quarterly and annual consolidated forecast across four statements:

- **Income Statement** (consolidated, GAAP and non-GAAP)
- **Balance Sheet** (assets, liabilities, equity, plus working-capital metrics)
- **Cash Flow** (operating, investing, financing; PP&E / depreciation schedules)
- **Segments** (revenue and gross-profit roll-up by product segment)

## Task

Using the assumptions provided in the **Operating Drivers** sheet, complete the forecast for the estimated periods marked in the workbook (columns labeled **\*E**). Populate all blank forecast cells across the Income Statement, Balance Sheet, Cash Flow, and Segments sheets, preserving the conventions, formulas, and cross-statement relationships established in the historical columns.

Key relationships to model:
- Derive segment-level revenue and gross profit from the operating-driver growth rates and margin assumptions; verify the roll-up matches consolidated totals.
- Build the Income Statement from top-line revenue through non-GAAP net income and per-share metrics.
- Flow the Balance Sheet forward each period, linking retained earnings to net income and dividends, PP&E to capex and depreciation, and ensuring the balance sheet balances (check row = 0).
- Construct the Cash Flow statement from net income with operating adjustments, capex, debt, equity transactions, and dividends; verify end-of-period cash matches the Balance Sheet.

## Deliverable

Save the completed workbook as:

    /app/output.xlsx

The verifier will re-calculate the workbook and check a set of key forecast cells.
