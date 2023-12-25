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

import os
from typing import Optional, Tuple, Union


def get_global_access_token() -> Optional[str]:
    return _get_val_from_env_var("EB_AGENT_ACCESS_TOKEN")


def get_global_save_dir() -> Optional[str]:
    return _get_val_from_env_var("EB_AGENT_SAVE_DIR")


def get_logging_level() -> Optional[str]:
    return _get_val_from_env_var("EB_AGENT_LOGGING_LEVEL")


def get_logging_file_path() -> Optional[str]:
    return _get_val_from_env_var("EB_AGENT_LOGGING_FILE")


def get_global_aksk() -> Tuple[Union[str, None], Union[str, None]]:
    return (_get_val_from_env_var("EB_AK"), _get_val_from_env_var("EB_SK"))


def _get_val_from_env_var(env_var: str) -> Optional[str]:
    return os.getenv(env_var)
