import dataclasses
import json
from typing import Dict, List, Optional, Union

data_path = (
    "/ssd2/tangshiyu/Code/ERNIE-Bot-SDK/erniebot-agent/benchmark/data/instruction_following_input_data.jsonl"
)


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: List[str]
    prompt: str
    kwargs: List[Dict[str, Optional[Union[str, int]]]]


def read_prompt_list(input_jsonl_filename):
    """Read inputs from jsonl."""
    inputs = []
    with open(input_jsonl_filename, "r") as file:
        for line in file:
            example = json.loads(line)
            breakpoint()

            inputs.append(
                InputExample(
                    key=example["key"],  # 序号
                    instruction_id_list=example["instruction_id_list"],
                    prompt=example["prompt"],
                    kwargs=example["kwargs"],
                )
            )
    return inputs


# build the data into the task format


if __name__ == "__main__":
    inputs = read_prompt_list(data_path)
