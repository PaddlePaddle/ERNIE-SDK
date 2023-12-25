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
from .config import GlobalConfig
from .config import init_global_config as _init_global_config
from .errors import ConfigItemNotFoundError as _ConfigItemNotFoundError
from .intro import Model
from .resources import (
    ChatCompletion,
    ChatCompletionResponse,
    ChatCompletionWithPlugins,
    Embedding,
    EmbeddingResponse,
    Image,
    ImageResponse,
    ImageV1,
    ImageV2,
)
from .response import EBResponse
from .utils.logging import setup_logging as _setup_logging
from .version import VERSION

__version__ = VERSION

__all__ = [
    "ChatCompletion",
    "ChatCompletionWithPlugins",
    "Embedding",
    "FineTuningTask",
    "FineTuningJob",
    "Image",
    "ImageV1",
    "ImageV2",
    "Model",
    "errors",
    "EBResponse",
    "ChatCompletionResponse",
    "EmbeddingResponse",
    "ImageResponse",
    "GlobalConfig",
    "__version__",
]

_init_global_config()

_setup_logging()


def __getattr__(name):
    # NOTE: We use a singleton to manage global configuration, which avoids some
    # of the pitfalls of setting global variables here (such as namespace
    # pollution and mutable global state) and further allows sanity checks.
    # Currently supported configuration options can be found in
    # erniebot/config.py.
    try:
        return GlobalConfig().get_value(name)
    except _ConfigItemNotFoundError:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from None
