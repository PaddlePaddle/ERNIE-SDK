import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from benchmark.tasks.task_funcs import TASK_DICT


@dataclass
class Report(object):
    """The report generate by each task."""

    task_name: str
    sucessful_score: float  # average successful score of the task
    stability: float  # the stability score of the task
    latency: float  # the average latency of the task
    task_weight: float  # the weight of the task when calculating the overall score

    def json(self):
        record_vars = ["task_name", "successful_score", "stability", "latency", "task_weight"]

        return str({var: getattr(self, var) for var in record_vars})


@dataclass
class Task(object):
    """The task that will be executed by the agent."""

    task_name: str
    task_id: int  # the id of the task in each task set
    task_weight: float
    prompt: str
    answer: str
    eval: Dict[str, Any]
    report: Optional[Report]

    async def run(self, agent, stability_trys: int = 10):
        """Run the task with the agent."""
        sucessful_score = []
        latencies = []
        correct_cnt = 0

        for i in range(stability_trys):
            breakpoint()
            output, latency = await self.run_once(agent)
            sucessfule_score = self.eval(output)  # varied from 0 to 100

            if sucessfule_score > 0:
                correct_cnt += 1
                latencies.append(latency)
            sucessful_score.append(sucessfule_score)

        self._build_report(
            correct_cnt / stability_trys, sum(sucessful_score) / correct_cnt, sum(latencies) / correct_cnt
        )

    async def run_once(self, agent):
        """Run the task once."""
        start = time.time()
        output = await agent.run(self.prompt)
        latency = time.time() - start
        return output, latency

    def eval_task(self, pred_output: str):
        """Evaluate the task."""
        if self.eval.eval_type == "exact":
            return (pred_output == self.answer) * 100
        elif self.eval.eval_type == "rough-L":
            pass
            # TODO: implement the rough-L method
        elif self.eval.eval_type == "LLM":
            pass
            # TODO: implement the LLM method
        elif (
            self.eval.eval_type == "func"
        ):  # function is a class, we can get result of alignment through the class func
            eval_funcs = self.eval.funcs
            following_res = {}
            for func in eval_funcs:
                func_name = func["name"]
                func_kwargs = func["kwargs"]
                is_following = TASK_DICT[func_name](**func_kwargs)(pred_output, self.answer)
                following_res[func_name] = is_following

            return following_res
            # TODO: implement the func method, expected output shoud get from the func
        else:
            raise NotImplementedError(
                "The eval method {} is not implemented. Only exact, rough-L, \
                    LLM and func is supported.".format(
                    self.eval.eval_type
                )
            )

    def _build_report(self, stability: float, sucessful_score: float, latency: float):
        self.report = Report(
            task_name=self.task_name,
            sucessful_score=sucessful_score,
            stability=stability,
            latency=latency,
            task_weight=self.task_weight,
        )


@dataclass
class AgentArgs(object):
    """The arguments of the agent."""

    agent: Dict[str, Any]
    model: Dict[str, Any]
    memory: Dict[str, Any]
    tools: Dict[str, Any]
    root_paths: Dict[str, str]
