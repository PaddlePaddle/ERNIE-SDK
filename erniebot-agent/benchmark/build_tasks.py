from typing import Any, Dict

from benchmark.schema import Task


class TaskBuilder(object):
    """The builder of the task."""

    def __init__(self, task_config: Dict[str, Any]):
        self.task_config = task_config

    def build_task(self) -> Task:
        """Build the task."""
        return Task(
            task_name=self.task_config["task_name"],
            task_id=self.task_config["task_id"],
            task_weight=self.task_config["task_weight"],
            prompt=self.task_config["prompt"],
            answer=self.task_config["answer"],
            eval=self.task_config["eval"],
            report=None,
        )
