#!/usr/bin/env python

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time

import erniebot

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    erniebot.api_type = "qianfan-sft"

    response = erniebot.FineTuningTask.create(name="sft_test", description="test")
    task_info = response.result
    print(task_info)

    train_config = {
        "epoch": 1,
        "batchSize": 4,
        "learningRate": 0.00003,
        "maxSeqLen": 4096,
    }
    train_set = [{"type": 2, "bosPath": "<path-to-dataset>"}]
    response = erniebot.FineTuningJob.create(
        task_id=task_info["id"],
        description="test",
        train_mode="SFT",
        peft_type="LoRA",
        train_config=train_config,
        train_set=train_set,
        train_set_rate=20,
    )
    job_info = response.result
    print(job_info)

    vdl_url = None
    while True:
        response = erniebot.FineTuningJob.query(task_id=task_info["id"], job_id=job_info["id"])
        running_info = response.result
        if vdl_url is None:
            vdl_url = running_info["vdlLink"]
            print(f"Check VisualDL logs at {vdl_url}")
        status = running_info["trainStatus"]
        if status == "RUNNING":
            print(f"Progress: {running_info['progress']}%")
            time.sleep(20)
            continue
        else:
            print(f"Status: {status}")
            break
