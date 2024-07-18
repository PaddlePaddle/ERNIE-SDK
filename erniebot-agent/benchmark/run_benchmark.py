import argparse
import asyncio
import sys
import time
from typing import List

import yaml

sys.path.append(".")

from benchmark.build_agent import AgentBuilder
from benchmark.build_tasks import TaskBuilder
from benchmark.schema import Report, Task
from benchmark.utils import recursively_search_yaml

from erniebot_agent.agents import Agent


class ReportAnalyzer:
    """
    write result to file.
    write following result to another file;
    print output to the console.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def init_reports(self, reports: List[Report]):
        self.reports = reports

    def write_full_report(self):
        """
        write full report to file.
        """
        time_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        output_path = "result/{}/full_report_{}.txt".format(self.agent_name, time_stamp)

        with open(output_path, "w") as f:
            for report in self.reports:
                print(report.json())
                f.write(report.json())
                f.write("\n")


class EBBenchmark:
    def __init__(self, args):
        self.agent_config_path = args.agent_config_path
        self.tasks_config_dir = args.tasks_config_dir
        self.agent = self._build_agent_from_config(self.agent_config_path)
        self.agent_name = ".".join(
            [self.agent_config["root_path"]["agent"], self.agent_config["agent"]["name"]]
        )
        self.tasks = self._build_tasks()
        self.analyzer = ReportAnalyzer(self.agent_name)

    def _build_agent_from_config(self, agent_config_path: str) -> Agent:
        with open(agent_config_path, "r") as f:
            self.agent_config = yaml.load(f, Loader=yaml.FullLoader)
        return AgentBuilder(self.agent_config).agent

    def _build_tasks(self) -> List[Task]:
        task_yaml_paths = recursively_search_yaml(self.tasks_config_dir)
        tasks = []
        for task_yaml_path in task_yaml_paths:
            with open(task_yaml_path, "r") as f:
                task_config = yaml.load(f, Loader=yaml.FullLoader)
                task = TaskBuilder(task_config).build_task()
                tasks.append(task)
                print(f"Task {task.task_name}_{task.task_id} loaded.")

        return tasks

    def run(self) -> List[Report]:
        reports = List[Report]
        for task in self.tasks:
            breakpoint()
            asyncio.run(task.run(self.agent))  # Agent show be initialized with full params
            reports.append(task.report)
            print(f"Task {task.name} finished.")

        self.analyzer.init_reports(reports=reports)
        self.analyzer.write_full_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent_config_path", type=str, default="./benchmark/configs/agents/default_agent.yaml"
    )
    parser.add_argument("--tasks_config_dir", type=str, default="./benchmark/tasks")
    args = parser.parse_args()

    benchmark = EBBenchmark(args)
    reports = benchmark.run()
