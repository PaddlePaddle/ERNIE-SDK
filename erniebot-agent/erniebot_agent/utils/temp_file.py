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

import atexit
import pathlib
import tempfile
from tempfile import TemporaryDirectory
from typing import Any, List

from erniebot_agent.utils.logging import logger

_tracked_temp_dirs: List[TemporaryDirectory] = []


def create_tracked_temp_dir(*args: Any, **kwargs: Any) -> pathlib.Path:
    # Borrowed from
    # https://github.com/pypa/pipenv/blob/247a14369d300a6980a8dd634d9060bf6f582d2d/pipenv/utils/fileutils.py#L197
    def _cleanup() -> None:
        try:
            temp_dir.cleanup()
        except Exception as e:
            logger.warning("Failed to clean up temporary directory: %s", temp_dir.name, exc_info=e)

    temp_dir = tempfile.TemporaryDirectory(*args, **kwargs)
    _tracked_temp_dirs.append(temp_dir)
    atexit.register(_cleanup)
    return pathlib.Path(temp_dir.name)
