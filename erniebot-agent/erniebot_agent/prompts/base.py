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
from abc import ABC, abstractmethod
from typing import List


class BasePromptTemplate(ABC):
    def __init__(self, input_variables: List[str]):
        self.input_variables: List[str] = input_variables
    
    @abstractmethod
    def format(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod    
    def format_prompt(self):
        raise NotImplementedError