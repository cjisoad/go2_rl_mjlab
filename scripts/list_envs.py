"""Script to list mjlab environments."""

import sys
from pathlib import Path

# Prefer this checkout over other editable projects also named src.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tyro
from prettytable import PrettyTable

import mjlab
import mjlab.tasks  # noqa: F401
import src.tasks
from mjlab.tasks.registry import list_tasks


def list_environments(keyword: str | None = None):
    """List all registered environments.

    Args:
      keyword: Optional filter to only show environments containing this keyword.
    """
    table = PrettyTable(["#", "Task ID"])
    table.title = "Tracking and Go2 Training Tasks"
    table.align["Task ID"] = "l"

    all_tasks = src.tasks.TASK_IDS
    idx = 0
    for task_id in all_tasks:
        if keyword and keyword.lower() not in task_id.lower():
            continue
        table.add_row([idx + 1, task_id])
        idx += 1

    print(table)
    if idx == 0:
        msg = "[INFO] No tasks matched"
        if keyword:
            msg += f" keyword '{keyword}'"
        print(msg)
    return idx


def main():
    return tyro.cli(list_environments, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
    main()
