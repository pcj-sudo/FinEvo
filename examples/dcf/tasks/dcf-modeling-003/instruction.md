# Aster Vale Products — Integrated Financial Model Completion

Complete the integrated financial model for Aster Vale Products. The workbook
contains a three-statement model with key assumptions, supporting calculation
schedules, and a DCF valuation. Two interconnected computation blocks are
missing and must be reconstructed.

## Your Task

### 1. D&A Roll-Forward Schedules (Calcs sheet)

The **Calcs** sheet contains depreciation and amortization schedules for PP&E
(rows 4–20) and intangible assets (rows 20–34). The 2020–2022 actual columns
(C–E) are complete. The forecast columns (2023–2027, columns F–J) need
formulas for the following rows:

- **PP&E beginning balance** (row 5) and **ending balance** (row 8)
- **Total depreciation expense** (row 18)
- **PP&E staggered depreciation breakdown** (rows 12–17), which are
  components of row 18:
  - Row 12: "of Existing PP&E" — annual depreciation of the 2022 ending
    balance, using the PP&E useful life from row 10.
  - Rows 13–17: "of New PP&E" — each row corresponds to one forecast-year
    vintage (rows 13, 14, 15, 16, 17 correspond to 2023, 2024, 2025, 2026,
    2027 capex respectively). Annual straight-line depreciation for each
    vintage is the year's capex divided by the useful life. Each vintage
    row is filled with that constant annual charge starting from the year
    the capex is incurred through the end of the forecast period.
- **Intangible assets beginning balance** (row 21) and **ending balance**
  (row 24)
- **Total amortization expense** (row 33)
- **IA staggered amortization breakdown** (rows 27–32), structured
  identically to the PP&E staggered breakdown (row 27: existing IA;
  rows 28–32: vintages 2023–2027), using the IA useful life from row 26.

### 2. DCF Valuation (DCF sheet)

The DCF sheet contains the discounted cash flow valuation framework. The
following computations are blank for all five forecast years (2023–2027) and
the summary rollup:

- **NOPAT** — net operating profit after tax (row 7)
- **D&A add-back** — depreciation and amortization for FCFF (row 8)
- **PP&E and intangible investment** — capex outflow (row 9)
- **Changes in NWC** — net working capital changes (row 10)
- **FCFF** — free cash flow to the firm (row 11)
- **Discount factors** — based on the WACC (row 13)
- **Discounted FCFF** — present value of each year's free cash flow (row 14)
- **Terminal value** and its present value (rows 16–18)
- **Enterprise value**, **equity value**, and **implied share price**
  (rows 20–26)

## Key Relationships

- The income statement's D&A total (row 17) is the sum of PP&E depreciation
  (Calcs row 18) and intangible amortization (Calcs row 33). This flows
  into operating income (IS row 20), which feeds the DCF NOPAT.
- PP&E depreciation uses a 9-year straight-line life (Calcs row 10).
  Intangible assets use a 7-year straight-line life (Calcs row 26).
- The sign convention used throughout: capex is a positive investment in
  Calcs (adds to ending balance), depreciation/amortization is negative
  (reduces ending balance). In the DCF, D&A is added back as a positive
  number to NOPAT, while capex is a negative outflow.
- WACC, perpetual growth rate, tax rate, and capital structure parameters
  are in the DCF sheet.

## Cross-Sheet Dependencies

The D&A roll-forward schedules on Calcs are upstream of the DCF computation:
they determine the D&A expense on the income statement, which flows into
operating income and ultimately NOPAT. Ensure both blocks are internally
consistent before computing the valuation.

Save your completed workbook as `/app/output.xlsx`.
