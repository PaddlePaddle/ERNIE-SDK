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

from typing import List, Tuple

__all__ = ["Model"]


class Model(object):
    """A dummy resource class."""

    @staticmethod
    def list() -> List[Tuple[str, str]]:
        """List the available models."""
        return [
            ("ernie-bot", "文心一言模型（ernie-bot）"),
            ("ernie-bot-turbo", "文心一言模型（ernie-bot-turbo）"),
            ("ernie-bot-4", "文心一言模型（ernie-bot-4）"),
            ("ernie-bot-8k", "文心一言模型（ernie-bot-8k）"),
            ("ernie-text-embedding", "文心百中语义模型"),
            ("ernie-vilg-v2", "文心一格模型"),
        ]
