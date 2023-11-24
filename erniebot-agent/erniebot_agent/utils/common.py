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

import requests


def get_cache_dir():
    """Use ~/.cache/erniebot_agent as the cache directory"""
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache", "erniebot_agent")
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    return cache_dir


def download_file(url: str, save_path: str):
    """Download file from url"""
    response = requests.get(url)
    assert response.status_code == 200, f"Download file failed: {url}."
    with open(save_path, "wb") as file:
        file.write(response.content)
