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

import argparse

import erniebot
from .config import GlobalConfig
from .errors import EBError
from .response import EBResponse
from .utils.logging import logger

__all__ = ['console_main', 'parse_args']


def console_main():
    args = parse_args()

    if args.verbose:
        logger.set_level('DEBUG')
    else:
        logger.set_level('INFO')

    cfg = GlobalConfig()

    if args.access_token:
        cfg.set_value('access_token', args.access_token)
    if args.ak:
        cfg.set_value('ak', args.ak)
    if args.sk:
        cfg.set_value('sk', args.sk)

    if args.api_base_url:
        cfg.set_value('base_url', args.api_base_url)
    if args.api_type:
        cfg.set_value('api_type', args.api_type)

    if args.proxy:
        cfg.set_value('proxy', args.proxy)
    if args.timeout:
        cfg.set_value('timeout', args.timeout)

    try:
        args.api_invoker(args)
    except EBError as e:
        logger.error("API invocation failed.", exc_info=e)
        return 1
    return 0


def parse_args(*args, **kwargs):
    parser = argparse.ArgumentParser(prog='erniebot')
    subparsers = parser.add_subparsers(dest='sub_command', required=True)

    # Global arguments
    parser.add_argument('--access-token', type=str, help="Access token to use.")
    parser.add_argument('--ak', type=str, help="API key or access key ID.")
    parser.add_argument(
        '--sk', type=str, help="Secret key or secret access key.")

    parser.add_argument('--api-base-url', type=str, help="API base URL.")
    parser.add_argument('--api-type', type=str, help="API type.")

    parser.add_argument('--proxy', type=str, help="Proxy to use.")
    parser.add_argument('--timeout', type=float, help="Timeout for retrying.")

    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help="Whether to increase verbosity.")
    parser.add_argument(
        '-V',
        '--version',
        action='version',
        version=f"%(prog)s {erniebot.__version__}",
        help="Show version number and exit.")

    subparser_api = subparsers.add_parser('api', help="API calls.")
    api_parsers = subparser_api.add_subparsers(dest='api', required=True)
    _register_resource(api_parsers, ChatCompletionHelper, 'chat_completion')
    _register_resource(api_parsers, ChatFileHelper, 'chat_file')
    _register_resource(api_parsers, ImageV2Helper, 'image')
    _register_resource(api_parsers, ModelHelper, 'model')

    return parser.parse_args(*args, **kwargs)


def _register_resource(subparsers, resource, parser_name_prefix):
    api_names = resource.get_api_names()
    for api_name in api_names:
        subparser = subparsers.add_parser(f"{parser_name_prefix}.{api_name}")
        subparser = resource.register_api_to_parser(subparser, api_name)


class _ResourceCLIHelper(object):
    @classmethod
    def add_resource_arguments(cls, parser):
        raise NotImplementedError

    @classmethod
    def get_api_names(cls):
        raise NotImplementedError

    @classmethod
    def get_resource_class(cls):
        raise NotImplementedError

    @classmethod
    def register_api_to_parser(cls, parser, api_name):
        def _find_method(method_name):
            if not hasattr(cls, method_name):
                raise AttributeError(
                    f"{cls.__name__}.{method_name} is not found.")
            method = getattr(cls, method_name)
            if not callable(method):
                raise TypeError(
                    f"{cls.__name__}.{method_name} is not callable.")
            return method

        cls.add_resource_arguments(parser)

        _add_args_method = _find_method(f"add_{api_name}_arguments")
        _add_args_method(parser)

        _invoke_api_method = _find_method(api_name)
        parser.set_defaults(api_invoker=_invoke_api_method)

        return parser


class ChatCompletionHelper(_ResourceCLIHelper):
    @classmethod
    def add_resource_arguments(cls, parser):
        return parser

    @classmethod
    def get_api_names(cls):
        return ['create']

    @classmethod
    def get_resource_class(cls):
        return erniebot.ChatCompletion

    # ChatCompletion.create
    @classmethod
    def add_create_arguments(cls, parser):
        parser.add_argument(
            '--message',
            type=str,
            action='append',
            nargs=2,
            required=True,
            metavar=('ROLE', 'CONTENT'),
            help="A message in `{role} {content}` format. This option can be specified multiple times to add multiple messages."
        )
        parser.add_argument(
            '--model', type=str, required=True, help="Model to use.")
        parser.add_argument(
            '--stream', action='store_true', help="Whether to stream messages.")
        parser.add_argument(
            '--temperature',
            type=float,
            help="Sampling temperature to use. A higher temperature encourages the model to generate more creative and varied content."
        )
        parser.add_argument(
            '--top-p',
            type=float,
            help="Parameter of nucleus sampling that affects the diversity of generated content. This option is better not used together with `--temperature`."
        )
        parser.add_argument(
            '--penalty-score',
            type=float,
            help="Penalty assigned to tokens that have been generated. A higher pernalty score reduces duplication in generated content."
        )
        parser.add_argument(
            '--user-id', type=str, help="Unique identifier of the user.")
        parser.add_argument(
            '--request-timeout',
            type=float,
            help="How many seconds to wait for the server to send data before giving up."
        )
        return parser

    @classmethod
    def create(cls, args):
        messages = [{
            'role': role,
            'content': content
        } for role, content in args.message]
        stream = args.stream
        kwargs = {'model': args.model, 'messages': messages, 'stream': stream}
        if args.temperature:
            kwargs['temperature'] = args.temperature
        if args.top_p:
            kwargs['top_p'] = args.top_p
        if args.penalty_score:
            kwargs['penalty_score'] = args.penalty_score
        if args.user_id:
            kwargs['user_id'] = args.user_id
        if args.request_timeout:
            kwargs['request_timeout'] = args.request_timeout
        resp = cls.get_resource_class().create(**kwargs)
        if isinstance(resp, EBResponse):
            responses = [resp]
        else:
            responses = resp
        for r in responses:
            logger.info(r.result)


class ChatFileHelper(_ResourceCLIHelper):
    @classmethod
    def add_resource_arguments(cls, parser):
        return parser

    @classmethod
    def get_api_names(cls):
        return ['create']

    @classmethod
    def get_resource_class(cls):
        return erniebot.ChatFile

    # ChatFile.create
    @classmethod
    def add_create_arguments(cls, parser):
        parser.add_argument(
            '--message',
            type=str,
            action='append',
            nargs=2,
            required=True,
            metavar=('ROLE', 'CONTENT'),
            help="A message in `{role} {content}` format. This option can be specified multiple times to add multiple messages."
        )
        parser.add_argument(
            '--stream', action='store_true', help="Whether to stream messages.")
        parser.add_argument(
            '--request-timeout',
            type=float,
            help="How many seconds to wait for the server to send data before giving up."
        )
        return parser

    @classmethod
    def create(cls, args):
        messages = [{
            'role': role,
            'content': content
        } for role, content in args.message]
        stream = args.stream
        kwargs = {'messages': messages, 'stream': stream}
        if args.request_timeout:
            kwargs['request_timeout'] = args.request_timeout
        resp = cls.get_resource_class().create(**kwargs)
        if isinstance(resp, EBResponse):
            responses = [resp]
        else:
            responses = resp
        for r in responses:
            logger.info(r.result)


class ImageV1Helper(_ResourceCLIHelper):
    @classmethod
    def add_resource_arguments(cls, parser):
        return parser

    @classmethod
    def get_api_names(cls):
        return ['create']

    @classmethod
    def get_resource_class(cls):
        return erniebot.ImageV1

    # ImageV1.create
    @classmethod
    def add_create_arguments(cls, parser):
        parser.add_argument(
            '--text',
            type=str,
            required=True,
            help="Text that describes the image(s).")
        parser.add_argument(
            '--resolution',
            type=str,
            required=True,
            help="Resolution of the image(s), e.g., '1024*1024'.")
        parser.add_argument(
            '--style', type=str, required=True, help="Art style to use.")
        parser.add_argument(
            '--num', type=int, help="Number of images to generate.")
        parser.add_argument(
            '--request-timeout',
            type=float,
            help="How many seconds to wait for the server to send data before giving up."
        )
        return parser

    @classmethod
    def create(cls, args):
        kwargs = {
            'text': args.text,
            'resolution': args.resolution,
            'style': args.style
        }
        if args.num:
            kwargs['num'] = args.num
        resp = cls.get_resource_class().create(**kwargs)
        logger.info("Image URLs:")
        for idx, item in enumerate(resp.data['imgUrls'], 1):
            logger.info("\tImage %d: %s", idx, item['image'])


class ImageV2Helper(_ResourceCLIHelper):
    @classmethod
    def add_resource_arguments(cls, parser):
        return parser

    @classmethod
    def get_api_names(cls):
        return ['create']

    @classmethod
    def get_resource_class(cls):
        return erniebot.ImageV2

    # ImageV1.create
    @classmethod
    def add_create_arguments(cls, parser):
        parser.add_argument(
            '--model', type=str, required=True, help="Model to use.")
        parser.add_argument(
            '--prompt',
            type=str,
            required=True,
            help="Text that describes the image(s).")
        parser.add_argument(
            '--width', type=int, required=True, help="Width of the image(s).")
        parser.add_argument(
            '--height', type=int, required=True, help="Height of the image(s).")
        parser.add_argument(
            '--version', type=str, help="Version of the model to use.")
        parser.add_argument(
            '--image-num', type=int, help="Number of images to generate.")
        parser.add_argument(
            '--request-timeout',
            type=float,
            help="How many seconds to wait for the server to send data before giving up."
        )
        return parser

    @classmethod
    def create(cls, args):
        kwargs = {
            'model': args.model,
            'prompt': args.prompt,
            'width': args.width,
            'height': args.height
        }
        if args.version:
            kwargs['version'] = args.version
        if args.image_num:
            kwargs['image_num'] = args.image_num
        resp = cls.get_resource_class().create(**kwargs)
        for idx_t, task_item in enumerate(resp.data['sub_task_result_list'], 1):
            logger.info("Subtask %d:", idx_t)
            status = task_item['sub_task_status']
            logger.info("Status: %s", status)
            assert status in ('SUCCESS', 'FAILED'), status
            if status == 'SUCCESS':
                logger.info("Image URLs:")
                for idx_i, image_item in enumerate(
                        task_item['final_image_list'], 1):
                    review_conclusion = image_item['img_approve_conclusion']
                    if review_conclusion == 'pass':
                        info = image_item['img_url']
                    elif review_conclusion == 'review':
                        info = "This image has been censored for possible policy violation."
                    elif review_conclusion == 'block':
                        info = "This image has been censored for policy violation."
                    else:
                        raise ValueError(
                            f"Invalid review conclusion: {review_conclusion}")
                    logger.info("\tImage %d: %s", idx_i, info)


class ModelHelper(_ResourceCLIHelper):
    @classmethod
    def add_resource_arguments(cls, parser):
        return parser

    @classmethod
    def get_api_names(cls):
        return ['list']

    @classmethod
    def get_resource_class(cls):
        return erniebot.Model

    # Model.list
    @classmethod
    def add_list_arguments(cls, parser):
        return parser

    @classmethod
    def list(cls, args):
        model_info_list = cls.get_resource_class().list()
        for model_name, model_desc in model_info_list:
            # XXX: Hard-code max name length
            logger.info("%24s %s", model_name, model_desc)
