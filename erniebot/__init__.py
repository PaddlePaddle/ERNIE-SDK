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

from . import errors
from .config import GlobalConfig, init_global_config
from .intro import Model
from .resources import (ChatCompletion, ChatFile, Embedding, FineTuningTask,
                        FineTuningJob, Image, ImageV1, ImageV2)
from .version import VERSION

__all__ = [
    'ChatCompletion',
    'ChatFile',
    'Embedding',
    'FineTuningTask',
    'FineTuningJob',
    'Image',
    'ImageV1',
    'ImageV2',
    'Model',
]

__version__ = VERSION

init_global_config()


def __getattr__(name):
    # NOTE: We use a singleton to manage global configuration, which avoids some
    # of the pitfalls of setting global variables here (such as namespace
    # pollution and mutable global state) and further allows sanity checks.
    # Currently supported configuration options can be found in
    # erniebot/config.py.
    return GlobalConfig().get_value(name)
