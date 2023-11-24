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

LOGGER_NAME: str = "erniebot"

STREAM_RESPONSE_PREFIX: bytes = b"data: "

DEFAULT_REQUEST_TIMEOUT_SECS: float = 600
MAX_CONNECTION_RETRIES: int = 2
MAX_SESSION_LIFETIME_SECS: float = 180

POLLING_INTERVAL_SECS: float = 5
POLLING_TIMEOUT_SECS: float = 20
