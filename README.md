# FinEvo

Code for **FinEvo: Self-Generated Challenging Tasks for Recursive Self-Improvement of Financial Spreadsheet Agents**.

FinEvo connects financial-workbook preparation, solver-guided task evolution and synchronous GRPO. A single target model constructs challenges, solves them in fresh contexts, reviews failed attempts and learns from executable rewards.

This repository includes one prepared DCF workbook and its ten accepted tasks: five modeling and five debugging. It includes the task-generation pipeline, standalone Harbor evaluation, and the synchronous GRPO implementation used by the paper. The example uses a fictional company and synthetic business values. Its 8/2 demonstration split is separate from the paper's 88/22 experiment.

```text
generation/          Same-model cleaning, construction, checks, hardening and review
evaluation/          Standalone Harbor evaluation and reporting
training/            Synchronous GRPO adapters, pinned backend and configurations
examples/dcf/        One prepared source workbook and ten frozen task packages
scripts/             Prepared-source registration and offline release validation
results/             Main paper statistics, without model or task histories
```

## Install

Use Python 3.12. Docker and LibreOffice are required for executing spreadsheet tasks; a model endpoint is required for generation and solver evaluation.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./generation -e ./evaluation -e './training[controller,dev]'
python -m pip install -r generation/requirements-openhands.txt
python scripts/validate_release.py
```

Read [generation/README.md](generation/README.md) for task generation, [evaluation/README.md](evaluation/README.md) for evaluation, and [training/README.md](training/README.md) for synchronous RL and its GPU prerequisites.

Run the CPU checks without calling a model or starting Docker/GPU work:

```bash
PYTHONPATH=generation/src:generation/tests:training/src:evaluation/src python -m pytest generation/tests training/tests -q
```

The supplied workbook is a prepared input. Its historical preparation sessions are not part of this release; the generic same-model cleaning and audit implementation remains available for new workbooks. The other ten source families, experiment histories, model weights and training checkpoints are not included.

The ten examples support inspecting and running the method. They do not reproduce the full paper's training result. [results/README.md](results/README.md) identifies the retained main-result summaries.

Third-party backend code retains its original license in [training/backend/slime/LICENSE](training/backend/slime/LICENSE).

The CPU CI template is provided in `ci/checks.yml`. To enable GitHub Actions, place it under `.github/workflows/` using credentials with workflow-management permission.
