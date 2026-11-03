# DCF example

This is the smallest source family by nonempty-cell and formula counts among the prepared workbook sources used in the study: six sheets, 1,675 nonempty cells and 1,143 formulas. It describes a fictional industrial company, including financial statements, depreciation/amortization, debt sweep and valuation.

- `reference.xlsx`: the complete prepared source workbook.
- `tasks/dcf-modeling-001` through `005`: five accepted modeling tasks.
- `tasks/dcf-debugging-001` through `005`: five accepted debugging tasks.
- `manifests/`: file hashes, prompts and an 8-task train / 2-task validation example split (task 005 of each type is validation).

Task instructions, worksheet XML, cached values and scoring code retain the original accepted content. Release task identifiers are shortened; document author/date properties are removed from XLSX metadata. This does not alter formulas or recalculate workbooks.

The split is for demonstrating the code on this ten-task subset, and does not replace the paper's experimental split. Reference workbooks and test targets belong to the private scoring side during execution, even though they are present in this inspectable repository.
