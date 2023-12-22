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

from typing import Any, List, Optional

from jinja2 import Environment, meta

from erniebot_agent.messages import HumanMessage
from erniebot_agent.prompt import BasePromptTemplate


def jinja2_formatter(template: str, **kwargs: Any) -> str:
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
    """
    Format the prompt for llm input.
    
    Args:
        template: The template string.
        name: The name of the prompt.
        input_variables: The input variables of the template.
    
    Attributes:
        name: The name of the prompt.
        template: The template string.
        input_variables: The input variables of the template.
        validate_template: Whether to validate the template.
    """

    def __init__(
        self, template: str, name: Optional[str] = None, input_variables: Optional[List[str]] = None
    ):
        super().__init__(input_variables)
        self.name = name
        self.template = template
        self.validate_template = True if input_variables is not None else False

    def format(self, **kwargs) -> str:
        """Fill the template with the given input variables."""
        if self.validate_template:
            error = self._validate_template()
            if error:
                raise KeyError("The input_variables of PromptTemplate and template are not match! " + error)
        return jinja2_formatter(self.template, **kwargs)

    def _validate_template(self):
        """
        Validate that the input variables are valid for the template.

        Args:
            template: The template string.
            input_variables: The input variables.
        """
        input_variables_set = set(self.input_variables)
        env = Environment()
        ast = env.parse(self.template)
        valid_variables = meta.find_undeclared_variables(ast)

        missing_variables = valid_variables - input_variables_set
        extra_variables = input_variables_set - valid_variables

        Error_message = ""
        if missing_variables:
            Error_message += f"The missing input variables: {missing_variables} "

        if extra_variables:
            Error_message += f"The extra input variables: {extra_variables}"

        return Error_message

    def format_as_message(self, **kwargs):
        """Return the prompt as a HumanMessage"""
        prompt = self.format(**kwargs)
        return HumanMessage(content=prompt)
