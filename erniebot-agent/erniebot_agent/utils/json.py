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

import json
from typing import Any


def to_compact_json(obj: Any, *, from_json: bool = False) -> str:
    if from_json:
        obj = json.loads(obj)
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def to_pretty_json(obj: Any, *, from_json: bool = False) -> str:
    if from_json:
        obj = json.loads(obj)
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, indent=2)
