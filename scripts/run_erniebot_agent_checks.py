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

import os
import subprocess
import sys


if __name__ == "__main__":
    cwd = "erniebot-agent"
    try:
        subprocess.check_call([sys.executable, os.path.join("scripts", "format_and_lint.py")], cwd=cwd)
        subprocess.check_call([sys.executable, os.path.join("scripts", "type_check.py")], cwd=cwd)
    except subprocess.CalledProcessError:
        sys.exit(1)
