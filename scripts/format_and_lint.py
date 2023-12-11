#!/usr/bin/env python

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import subprocess
import sys

DEFAULT_FILES = (
    "src",
    "examples",
    "tests",
    "scripts",
)


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = DEFAULT_FILES
    run_black(files)
    run_isort(files)
    run_flake8(files)


def run_black(files):
    print("===== black =====")
    subprocess.check_call([sys.executable, "-m", "black", *files])


def run_flake8(files):
    print("===== flake8 =====")
    subprocess.check_call([sys.executable, "-m", "flake8", *files])


def run_isort(files):
    print("===== isort =====")
    subprocess.check_call([sys.executable, "-m", "isort", "--filter-files", *files])


if __name__ == "__main__":
    main()
