# Main-result summaries

`rl-checkpoints.csv` contains the main development learning curve through update 8. Each row averages five independent attempts on each of 22 tasks (110 attempts), not best-of-five. Retaining the original failures in two retried slots gives 28.18% task success. These are historical paper summaries, not results on the ten released examples.

`generation-rounds.csv` contains the ten content-round counts behind the paper's generation-progress figure. `current_valid` equals cumulative acceptance plus the current round's hardening count. It does not accumulate old hardening events.

No earlier training pilots, RFT results, trajectories or per-task histories are included. The complete historical evidence remains in the research archive outside this release.
