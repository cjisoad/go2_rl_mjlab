# Remote training

The source checkout is `/mjlab/unitree_rl_mjlab`, with a dedicated Conda environment at `/mjlab/envs/go2`. Logs, status and checkpoints live under the checkout's `logs/` directory, on persistent storage. Stop/recreate the server only after confirming the `/mjlab` storage remains attached.

## Script 1: sequential training (server)

```bash
cd /mjlab/unitree_rl_mjlab
mkdir -p logs/sequence
nohup /mjlab/envs/go2/bin/python -u scripts/train_sequence.py \
  > logs/sequence/launcher.log 2>&1 < /dev/null &
```

The order and registry defaults are frozen when the queue starts:

| Task | Iterations | Environments |
| --- | ---: | ---: |
| Flat | 2000 | 4096 |
| Rough | 3000 | 4096 |
| Backflip V1.6 | 1000 | 4096 |
| StandUp V2.1 | 6000 | 4096 |
| DreamWaQ | 5000 | 4096 |
| AMP | 10000 | 2048 |
| Quad Pedipulation | 5000 | 4096 |
| Biped Pedipulation | 6000 | 4096 |

Only one queue can hold the state-directory lock. A failed task stops the queue and keeps its log. Each task's PPO stdout is under `logs/sequence/<queue_id>/`; models and TensorBoard events remain under `logs/rsl_rl/<experiment>/<timestamp_queue_id>/`. `logs/sequence/status.json` records their exact paths. Completed tasks include first/last 100 reward means, best reward and last checkpoint. Reward scales differ across tasks; these summaries describe training trends, not success rates or verified convergence.

Stop gracefully with `kill -TERM <pid>` using the queue PID in status.json. `--resume` skips completed tasks and restarts the unfinished task **from scratch**, retaining old logs and model runs. This is queue resumption, not checkpoint continuation. For a completely new queue use a different `--state-dir` and pass its status path to the monitor.

## Script 2: status query (local WSL)

```bash
cd /home/e980/go2_rl_mjlab
/home/e980/miniforge3/envs/cjisoad_mjlab/bin/python scripts/monitor_remote.py
# Refresh every 30 seconds; enter the SSH password once per invocation.
/home/e980/miniforge3/envs/cjisoad_mjlab/bin/python scripts/monitor_remote.py --watch 30
```

Only Paramiko is needed for querying (`python -m pip install paramiko==5.0.0`); no local GPU or mjlab is required. The script also runs with Windows Python if Paramiko is installed. Endpoint defaults: root@xj-member.bitahub.com:42305. It prompts for the password without saving it. `--key PATH` uses a private key instead; `--host`, `--port`, `--user`, `--state`, `--json` override query settings. Host keys are persisted in `~/.ssh/known_hosts` on first contact; later changes are rejected.

The query displays the current task, iteration/target, environments, latest and first/last reward means, episode length, ETA and model paths for previous tasks. It checks whether the queue process is still alive and marks a stale running state as stopped. SSH errors produce a nonzero exit instead of displaying cached progress. Ctrl+C exits watch mode without stopping training.

## Recreate the server environment

The server image supplies PyTorch 2.8.0+cu126 and CUDA runtime libraries. The dedicated environment preserves the image CUDA/PyTorch packages. Since the image has no Conda package cache, Python and pip were installed into the copied environment before installing project dependencies:

```bash
/opt/conda/bin/conda create -p /mjlab/envs/go2 --clone /opt/conda -y
# If the source image has no package cache, complete the Python runtime:
/opt/conda/bin/conda create -p /mjlab/envs/go2 python=3.11 pip -y --override-channels -c conda-forge
/mjlab/envs/go2/bin/python -m pip install -r requirements-training.txt
/mjlab/envs/go2/bin/python -m pip install -e . --no-deps
```

Use a CUDA-enabled PyTorch 2.8.0 build if recreating on another image. The original source archive and `docs/migration/deployment-files.json` record the transferred snapshot. No private keys, passwords or local training logs are included.

## Append tasks to a queue that is already running

The current six-task process has its original task list in memory. A separate
`scripts/extend_training_queue.py` process waits for its queue lock, requires
successful completion, then launches `train_sequence.py --resume`. Resume accepts
an existing prefix of the eight-task order and preserves all completed task
summaries and checkpoints. It appends only the missing tasks. Failure of the
original queue stops the handoff as well.

The handoff writes `logs/sequence/planned_tasks.json`; the monitor merges these
pending entries into its display until they appear in the active status file.
The deployed handoff is detached from SSH and logs to
`logs/sequence/extension.log`. Future new queues contain all eight tasks directly.
