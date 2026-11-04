# Evaluation

From the repository root, set `LLM_BASE_URL` and `LLM_API_KEY` for the OpenAI-compatible model endpoint (see the root README):

```bash
rsi-eval --root evaluation --config configs/openhands.json --job-name financialstatements1-example --model openai/Qwen3.6-27B
```

The config selects both `examples/financialstatements1/tasks` and `examples/financialstatements2/tasks`, five independent attempts per task, OpenHands SDK 1.47.0, 75 agent iterations and 1,800 seconds per attempt. Configure the model endpoint through environment variables. Use a fresh job name on each invocation.

`configs/oracle.json` runs the reference solution for an optional environment/scorer check. Every package contains `instruction.md`, public workbook input, a solution reference, private targets, the fixed verifier and a Docker environment. Harbor keeps solution and test files out of the solver's public environment. The reward is one only when all target cells are correct after recalculation; outside-target differences are diagnostic.

These commands execute Docker tasks and, for the solver configuration, call the configured model. Offline repository validation does neither.
