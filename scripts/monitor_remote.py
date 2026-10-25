"""Query the remote Go2 training queue; needs only Python and paramiko locally."""
from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import shlex
import sys
import time


def connect(args):
    import paramiko
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    known = Path.home() / ".ssh" / "known_hosts"
    known.parent.mkdir(parents=True, exist_ok=True)
    if known.exists():
        client.load_host_keys(str(known))
    # Trust on first use, persist the host key, reject subsequent changes.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    password = None if args.key else getpass.getpass("SSH password: ")
    client.connect(args.host, port=args.port, username=args.user,
                   password=password, key_filename=str(args.key) if args.key else None,
                   look_for_keys=bool(args.key), allow_agent=bool(args.key), timeout=20)
    client.save_host_keys(str(known))
    return client



def merge_planned_tasks(state, planned):
    existing = {task["task"] for task in state["tasks"]}
    state["tasks"].extend(task for task in planned.get("tasks",[]) if task["task"] not in existing)

def query(client, args):
    # This server's persistent filesystem invalidates an open SFTP handle when
    # status.json is replaced. Read the snapshot entirely on the server instead.
    for attempt in range(3):
        _, stdout, stderr = client.exec_command(f"cat -- {shlex.quote(args.state)}", timeout=15)
        content = stdout.read()
        error = stderr.read().decode(errors="replace").strip()
        if stdout.channel.recv_exit_status() == 0:
            state = json.loads(content)
            break
        if attempt == 2:
            raise RuntimeError(f"Cannot read queue status: {error}")
        time.sleep(0.2)
    planned_path = shlex.quote(str(Path(args.state).with_name("planned_tasks.json")))
    _, out, err = client.exec_command(f"if [ -f {planned_path} ]; then cat -- {planned_path}; fi", timeout=15)
    planned = out.read()
    if out.channel.recv_exit_status() != 0:
        raise RuntimeError(err.read().decode(errors="replace"))
    if planned:
        merge_planned_tasks(state, json.loads(planned))
    # A stale JSON file must not imply that the training process is still alive.
    pid = int(state["pid"])
    command = f"cat /proc/{pid}/cmdline"
    _, stdout, stderr = client.exec_command(command, timeout=15)
    command_line = stdout.read().decode(errors="replace")
    alive = stdout.channel.recv_exit_status() == 0 and "train_sequence.py" in command_line
    state["queue_process_alive"] = alive
    if state["status"] == "running" and not alive:
        state["status"] = "stopped_without_final_status"
    # Checkpoints may have been saved since the last status write.
    for task in state["tasks"]:
        if task.get("status") == "running" and task.get("experiment_dir"):
            directory = shlex.quote(task["experiment_dir"])
            _, out, _ = client.exec_command(f"ls -t {directory}/model_*.pt 2>/dev/null | head -1")
            latest = out.read().decode().strip()
            if latest:
                task["checkpoint"] = latest
    return state


def number(value):
    return "—" if value is None else f"{value:.2f}"


def display(state):
    names = {"pending": "等待", "running": "训练中", "completed": "完成",
             "failed": "失败", "interrupted": "中断"}
    print(f"\n队列：{state['status']} | 进程存活：{state['queue_process_alive']} | 更新：{state['updated_at']}")
    print(f"当前任务：{state.get('current_task') or '无'}")
    for task in state["tasks"]:
        completed = task.get("completed_iterations", 0)
        target = task["max_iterations"]
        print(f"  {task['task']:<27} {names.get(task['status'], task['status']):<5} "
              f"{completed:>5}/{target:<5} ({100*completed/target:5.1f}%)  并行={task['num_envs']}")
        if "reward" in task:
            print(f"    奖励：最近={number(task.get('reward'))}  "
                  f"前100次均值={number(task.get('reward_first_100_mean'))}  "
                  f"末100次均值={number(task.get('reward_last_100_mean'))}  "
                  f"最佳={number(task.get('reward_best'))}  回合步数={number(task.get('episode_length'))}")
        if task["status"] == "running":
            print(f"    预计剩余：{task.get('eta', '编译或初始化中')}  "
                  f"最近输出：{task.get('last_message', '等待训练输出')}")
        if task.get("checkpoint"):
            print(f"    模型：{task['checkpoint']}")
        if task["status"] in ("failed", "interrupted"):
            print(f"    退出码：{task.get('return_code')}  日志：{task.get('log_file')}")
    if state.get("error"):
        print("错误：", state["error"])
    print("奖励摘要用于观察训练趋势；不同任务的奖励数值不能直接比较，完成轮次也不代表策略达标。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="xj-member.bitahub.com")
    parser.add_argument("--port", type=int, default=42305)
    parser.add_argument("--user", default="root")
    parser.add_argument("--key", type=Path)
    parser.add_argument("--state", default="/mjlab/unitree_rl_mjlab/logs/sequence/status.json")
    parser.add_argument("--watch", type=float, default=0, metavar="SECONDS",
                        help="Refresh interval; zero queries once")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.watch < 0:
        parser.error("--watch must be nonnegative")
    client = connect(args)
    try:
        while True:
            state = query(client, args)
            if args.json:
                print(json.dumps(state, ensure_ascii=False, indent=2))
            else:
                display(state)
            if not args.watch or state["status"] != "running":
                break
            time.sleep(args.watch)
    finally:
        client.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"查询失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
