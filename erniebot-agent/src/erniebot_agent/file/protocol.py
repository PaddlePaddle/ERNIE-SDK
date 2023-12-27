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

import datetime
import re
from typing import Generator, List, Literal, get_args

from typing_extensions import TypeAlias

_LOCAL_FILE_ID_PREFIX = "file-local-"
_UUID_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_LOCAL_FILE_ID_PATTERN = _LOCAL_FILE_ID_PREFIX + _UUID_PATTERN
_REMOTE_FILE_ID_PREFIX = "file-"
_REMOTE_FILE_ID_PATTERN = _REMOTE_FILE_ID_PREFIX + r"[0-9]{15}"

FilePurpose: TypeAlias = Literal["assistants", "assistants_output"]

_compiled_local_file_id_pattern = re.compile(_LOCAL_FILE_ID_PATTERN)
_compiled_remote_file_id_pattern = re.compile(_REMOTE_FILE_ID_PATTERN)


def create_local_file_id_from_uuid(uuid: str) -> str:
    """Create a random local file id."""
    return _LOCAL_FILE_ID_PREFIX + uuid


def get_timestamp() -> str:
    return datetime.datetime.now().isoformat(sep=" ", timespec="seconds")


def is_file_id(str_: str) -> bool:
    """Judge whether a file id is valid or not."""
    return is_local_file_id(str_) or is_remote_file_id(str_)


def is_local_file_id(str_: str) -> bool:
    """Judge whether a file id is a valid local file id or not."""
    return _compiled_local_file_id_pattern.fullmatch(str_) is not None


def is_remote_file_id(str_: str) -> bool:
    """Judge whether a file id is a valid remote file id or not."""
    return _compiled_remote_file_id_pattern.fullmatch(str_) is not None


def extract_file_ids(str_: str) -> List[str]:
    """Find all file ids in a string."""
    return extract_local_file_ids(str_) + extract_remote_file_ids(str_)


def extract_local_file_ids(str_: str) -> List[str]:
    """Find all local file ids in a string."""
    return _compiled_local_file_id_pattern.findall(str_)


def extract_remote_file_ids(str_: str) -> List[str]:
    """Find all remote file ids in a string."""
    return _compiled_remote_file_id_pattern.findall(str_)


def is_valid_file_purpose(file_purpose: str) -> bool:
    return file_purpose in get_args(FilePurpose)


def generate_fake_remote_file_ids() -> Generator[str, None, None]:
    counter = 0
    while True:
        number = f"{counter:015d}"
        if len(number) > 15:
            break
        yield _REMOTE_FILE_ID_PREFIX + number
