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


class SingletonMeta(type):
    _insts: dict = {}

    def __call__(cls, *args, **kwargs):
        # XXX: We note that the instance created in this way can be actually
        # copied by `copy.copy` or `copy.deepcopy`, which results in multiple
        # instances. Perhaps we should forbid the copy operations by patching
        # the created instance.
        if cls not in cls._insts:
            if cls not in cls._insts:
                cls._insts[cls] = super().__call__(*args, **kwargs)
        return cls._insts[cls]
