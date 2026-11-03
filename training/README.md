# Synchronous GRPO

This package retains the paper's binary-reward GRPO path: fresh current-policy trajectories, group-relative normalization, no KL or entropy penalty, and a barrier between sampling and each parameter update. Historical training recipes and their deployment scripts are omitted.

## Runtime

Python 3.12 is required for the CPU Harbor/OpenHands controller. The GPU path requires the pinned Slime/SGLang/Megatron environment recorded in `locks/stack.json`. The actual runtime Slime packages, including the model and agent adaptations used by this recipe, are vendored under `backend/slime`; their Apache-2.0 license is preserved. Install GPU dependencies in the pinned runtime container, and set `MEGATRON_ROOT` to its Megatron-LM checkout. No GPU image archives, weights or optimizer checkpoints are bundled.

The retained distributed launcher uses six nodes with eight GPUs per node: two learner nodes (TP8/DP2) and four dedicated rollout nodes (TP2 engines). Node addresses and reachable interfaces are supplied through environment variables. No original cluster launcher, allocation IDs or internal addresses are included.

## Prepare data and controller

From the repository root, with the local packages installed:

```bash
python scripts/validate_release.py
python scripts/prepare_training_schedule.py --profile training/configs/demo.json
docker build -t finevo-agent-runtime training/runtime
python scripts/configure_controller.py \
  --agent-image-id "$(docker image inspect --format '{{.Id}}' finevo-agent-runtime)" \
  --gateway-url http://learner-gateway-address:18001
mkdir -p private
python -c "from pathlib import Path; import secrets; p=Path('private/controller-token'); p.write_text(secrets.token_urlsafe(32)); p.chmod(0o600)"
rsi-train --root examples/dcf serve --config private/controller.json \
  --token-file private/controller-token --host 0.0.0.0 --port 18002
```

Set `gateway_url` in a local copy of the controller configuration to the learner gateway address reachable from the CPU and Docker tasks. The controller's data root must refer to the same physical example tasks and generated artifacts as the GPU-side `RSI_DATASET_ROOT` mount.

## GPU launch

Run inside the prepared GPU runtime on each node, with the repository and task data mounted at a shared path. Set these variables to your deployment:

```bash
export RSI_ROOT=/path/to/FinEvo
export RSI_RUN_ID=dcf-demo
export RSI_NODE_IPS=learner0,learner1,rollout0,rollout1,rollout2,rollout3
export RSI_GPU_IP=learner0
export RSI_LOCAL_IP=this-node-address
export RSI_NODE_RANK=0  # set 0 through 5 on the respective nodes
export RSI_TRAIN_PROFILE=$RSI_ROOT/training/configs/demo.json
export RSI_OPTIMIZATION_CONFIG=$RSI_ROOT/training/configs/optimization.json
export RSI_CONTROLLER_URL=http://controller-address:18002
export RSI_CONTROLLER_TOKEN_FILE=$RSI_ROOT/private/controller-token
export RSI_DATASET_ROOT=$RSI_ROOT/examples/dcf
export NCCL_SOCKET_IFNAME='=your-interface'
export GLOO_SOCKET_IFNAME=your-interface
export MEGATRON_ROOT=/opt/Megatron-LM
bash "$RSI_ROOT/training/tools/train_gpu.sh" train
```

Place the authorized starting weights at `models/Qwen3.6-27B-hf`; the revision is recorded in `locks/stack.json`. Non-head nodes join the head and wait under a bounded lease. Stop worker containers through your orchestration after head completion; `runs/<run-id>/stop.json` also ends their wait. Network-isolated containers are recommended so each run owns its Ray processes.

The two-update demo uses four distinct queries per update and eight trajectories per query from the supplied 8/2 example split. It retains the distributed GPU topology, so it is a data-size example rather than a low-resource training recipe. `configs/paper.json` records the eight-update 32×8 configuration. It requires an authorized complete dataset (88 training / 22 development tasks), supplied via `RSI_DATASET_ROOT`, and a schedule generated with `scripts/prepare_training_schedule.py --profile training/configs/paper.json`. That full dataset and the original screened first-batch selection are not bundled; a new schedule is a fresh run, not a replay of the paper's training path.

Learning rate is 4e-6, Adam betas (0.9, 0.98), weight decay 0.1 and clipping 0.2. Context limit is 262,144 and the retained response limit is 16,384 tokens. The reported run used 8,192 for its first two updates; see `results/rl-checkpoints.csv`. The release uses the final 16K implementation as its default. Tools/condensation outputs are excluded from the solver token loss.

CPU tests check scheduling, rewards and artifact handling. This curated release has not been launched for a new GPU training run.
