# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import io
import os

import setuptools


def read_file(filepath):
    with open(filepath) as fin:
        requirements = fin.read()
    return requirements


__version__ = read_file("VERSION")


def write_version_py(filename="erniebot_agent/version.py"):
    cnt = """# THIS FILE IS GENERATED FROM ERNIEBOT-AGENT SETUP.PY
VERSION           = '%(version)s'
"""
    content = cnt % {"version": __version__}

    with open(filename, "w") as f:
        f.write(content)


REQUIRED_PACKAGES = read_file("./requirements.txt")


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


setuptools.setup(
    name="erniebot-agent",
    version=__version__,
    author="PaddleNLP Team",
    author_email="paddlenlp@baidu.com",
    description="Easy-to-use and ErnieBot Agent",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/PaddlePaddle/ERNIE-Bot-Agent",
    license_files=("LICENSE", ),
    packages=setuptools.find_packages(
        where=".",
        exclude=("examples*", "tests*", "docs*", "cookbook*"),
    ),
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache 2.0",
)
