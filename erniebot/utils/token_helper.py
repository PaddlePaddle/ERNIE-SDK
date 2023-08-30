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

import math
import re

__all__ = ['approx_num_tokens']


def approx_num_tokens(text: str) -> int:
    """Estimate the number of tokens for a given piece of text."""
    cnt_han = 0
    cnt_word = 0

    res = []
    for char in text:
        if re.match(r'[\u4e00-\u9fff]', char):
            cnt_han += 1
            res.append(' ')
        elif re.match(r'[^\w\s]', char):
            res.append(' ')
        else:
            res.append(char)

    res_text = ''.join(res)
    cnt_word = len(res_text.split())

    return cnt_han + int(math.floor(cnt_word * 1.3))
