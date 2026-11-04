# financialstatements2

A prepared fictional financial-statements workbook with seven connected worksheets, quarterly and annual statements, operating drivers and segment schedules.

- `reference.xlsx`: the complete prepared source workbook.
- `tasks/financialstatements2-modeling-001` through `005`: five accepted modeling tasks.
- `tasks/financialstatements2-debugging-001` through `005`: five accepted debugging tasks.
- `manifests/`: frozen task hashes, public prompts and an 8-task train / 2-task validation demonstration split (task 005 of each type is validation).

Task instructions, worksheet XML, cached values and scoring code retain the accepted content. Release identifiers are standardized; document author/date properties are removed from XLSX metadata. Formulas are not changed and workbooks are not recalculated.

The demonstration split does not replace the paper's experimental split. References and scoring targets are private during solver execution, even though they are included for inspection in this repository.
