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

from erniebot_agent.prompts import BasePromptTemplate


def jinja2_formatter(template: str, **kwargs: any) -> str:
    """Format a template using jinja2."""
    try:
        from jinja2 import Template
    except ImportError:
        raise ImportError(
            "jinja2 not installed, which is needed to use the jinja2_formatter. "
            "Please install it with `pip install jinja2`."
        )

    return Template(template).render(**kwargs)
    
class PromptTemplate(BasePromptTemplate):
    """format the prompt for llm input."""
    def __init__(self, template, input_variables, name=None):
        super().__init__(input_variables)
        self.name = name
        self.template = template
        self.validate_template=None # todo，评估template中的合法性，langchain中评估变量是否符合预期 yes
        
    def format(self, **kwargs):
        jinja2_formatter(self.template, **kwargs)

    def format_prompt(self,): # todo：确定是否需要，用于转换prompt为str/Message。 yes to user message
        raise NotImplementedError('format_prompt is not implemented yet.')