#!/usr/bin/env python

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
import collections
import copy
import functools
import inspect
import json
import operator
import reprlib
import textwrap
import time
import traceback

import gradio as gr

import erniebot as eb

CUSTOM_FUNC_NAME = "custom_function"
MAX_CONTEXT_LINES_TO_SHOW = 10


def parse_setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8073)
    args = parser.parse_args()
    return args


def create_ui_and_launch(args):
    with gr.Blocks(
        title="ERNIE Bot SDK Function Calling Demo", theme=gr.themes.Soft(spacing_size="sm", text_size="md")
    ) as block:
        gr.Markdown("# ERNIE Bot SDK函数调用功能演示")
        create_components(functions=get_predefined_functions())

    block.queue(api_open=False, concurrency_count=8).launch(server_name="0.0.0.0", server_port=args.port)


def create_components(functions):
    func_name_list = list(map(operator.attrgetter("name"), functions))
    name2function = collections.OrderedDict(zip(func_name_list, functions))
    default_state = {"name2function": name2function, "context": []}
    default_api_type = "qianfan"
    default_model = "ernie-bot"

    state = gr.State(value=default_state)
    auth_state = gr.State(
        value={
            "api_type": default_api_type,
            "ak": "",
            "sk": "",
            "access_token": "",
        }
    )

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion(label="基础配置", open=True):
                        with gr.Group():
                            api_type = gr.Dropdown(
                                label="API Type",
                                info="提供对话能力的后端平台",
                                value=default_api_type,
                                choices=["qianfan", "aistudio"],
                            )
                            access_key = gr.Textbox(
                                label="AK",
                                info="用于访问后端平台的API key或access key ID",
                                type="password",
                                visible=(default_api_type == "qianfan"),
                            )
                            secret_key = gr.Textbox(
                                label="SK",
                                info="用于访问后端平台的secret key或secret access key",
                                type="password",
                                visible=(default_api_type == "qianfan"),
                            )
                            access_token = gr.Textbox(
                                label="Access Token", info="用于访问后端平台的access token", type="password"
                            )
                            ernie_model = gr.Dropdown(
                                label="Model", info="模型类型", value=default_model, choices=["ernie-bot"]
                            )
                    with gr.Accordion(label="高级配置", open=False):
                        with gr.Group():
                            top_p = gr.Slider(
                                label="Top-p",
                                info="控制采样范围，该参数越小生成结果越稳定",
                                value=0.7,
                                minimum=0,
                                maximum=1,
                                step=0.05,
                            )
                            temperature = gr.Slider(
                                label="Temperature",
                                info="控制采样随机性，该参数越小生成结果越稳定",
                                value=0.95,
                                minimum=0.05,
                                maximum=1,
                                step=0.05,
                            )

                with gr.Column(scale=3):
                    context_chatbot = gr.Chatbot(
                        label="对话历史",
                        latex_delimiters=[
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": "$", "right": "$", "display": False},
                        ],
                        bubble_full_width=False,
                    )
                    input_text = gr.Textbox(label="消息内容", value="请问12和16的“魔法运算”结果是多少？", placeholder="请输入...")
                    with gr.Row():
                        send_text_btn = gr.Button("发送消息")
                        regen_btn = gr.Button("重新生成")
                    with gr.Row():
                        recall_btn = gr.Button("撤回消息")
                        clear_btn = gr.Button("重置对话")

            func_call_accord = gr.Accordion(label="函数调用", open=False)
            with func_call_accord:
                chosen_func_names = gr.CheckboxGroup(
                    label="备选函数", value=func_name_list, choices=func_name_list + [CUSTOM_FUNC_NAME]
                )

                with gr.Row():
                    with gr.Tabs():
                        for function in functions:
                            create_function_tab(function)
                        with gr.Tab(label="自定义函数"):
                            custom_func_code = gr.Code(
                                label="定义",
                                value=get_custom_func_def_template(),
                                language="python",
                                interactive=True,
                            )
                            update_func_desc_btn = gr.Button("更新描述")
                            custom_func_desc = JSONCode(
                                label="描述",
                                value=to_pretty_json(get_custom_func_desc_template(), from_json=False),
                                interactive=True,
                            )

                    with gr.Column(scale=1):
                        func_name = gr.Textbox(label="函数名称")
                        func_in_params = JSONCode(label="请求参数")
                        func_out_params = JSONCode(label="响应参数", interactive=False)
                        with gr.Row():
                            call_func_btn = gr.Button("调用函数")
                            send_res_btn = gr.Button("发送调用结果")
                        reset_func_btn = gr.Button("重置函数调用信息")

        with gr.Accordion(label="原始对话上下文信息", open=False):
            raw_context_json = gr.JSON(label=f"最近{MAX_CONTEXT_LINES_TO_SHOW}条消息", scale=1)

    api_type.change(
        update_api_type,
        inputs=[
            auth_state,
            api_type,
        ],
        outputs=[
            auth_state,
            access_key,
            secret_key,
        ],
    ).success(
        reset_conversation,
        inputs=state,
        outputs=[
            state,
            context_chatbot,
            input_text,
            raw_context_json,
            func_name,
            func_in_params,
            func_out_params,
        ],
    )
    access_key.change(
        make_state_updater(key="ak", strip=True),
        inputs=[
            auth_state,
            access_key,
        ],
        outputs=auth_state,
    )
    secret_key.change(
        make_state_updater(key="sk", strip=True),
        inputs=[
            auth_state,
            secret_key,
        ],
        outputs=auth_state,
    )
    access_token.change(
        make_state_updater(key="access_token", strip=True),
        inputs=[
            auth_state,
            access_token,
        ],
        outputs=auth_state,
    )

    disable_chat_input_args = {
        "fn": lambda: replicate_gradio_update(6, interactive=False),
        "inputs": None,
        "outputs": [
            input_text,
            clear_btn,
            recall_btn,
            regen_btn,
            send_text_btn,
            send_res_btn,
        ],
        "show_progress": False,
        "queue": False,
    }
    enable_chat_input_args = {
        "fn": lambda: replicate_gradio_update(6, interactive=True),
        "inputs": None,
        "outputs": [
            input_text,
            clear_btn,
            recall_btn,
            regen_btn,
            send_text_btn,
            send_res_btn,
        ],
        "show_progress": False,
        "queue": False,
    }
    input_text.submit(**disable_chat_input_args).then(
        generate_response_for_text,
        inputs=[
            state,
            chosen_func_names,
            auth_state,
            ernie_model,
            input_text,
            top_p,
            temperature,
        ],
        outputs=[
            state,
            context_chatbot,
            input_text,
            raw_context_json,
            func_name,
            func_in_params,
            func_out_params,
            func_call_accord,
        ],
        show_progress=False,
    ).then(**enable_chat_input_args)
    send_text_btn.click(**disable_chat_input_args).then(
        generate_response_for_text,
        inputs=[
            state,
            chosen_func_names,
            auth_state,
            ernie_model,
            input_text,
            top_p,
            temperature,
        ],
        outputs=[
            state,
            context_chatbot,
            input_text,
            raw_context_json,
            func_name,
            func_in_params,
            func_out_params,
            func_call_accord,
        ],
        show_progress=False,
    ).then(**enable_chat_input_args)
    regen_btn.click(**disable_chat_input_args).then(
        lambda history: (history and history[:-1], gr.update(interactive=False)),
        inputs=context_chatbot,
        outputs=[
            context_chatbot,
            regen_btn,
        ],
        show_progress=False,
        queue=False,
    ).then(
        regenerate_response,
        inputs=[
            state,
            chosen_func_names,
            auth_state,
            ernie_model,
            top_p,
            temperature,
        ],
        outputs=[
            state,
            context_chatbot,
            input_text,
            raw_context_json,
            func_name,
            func_in_params,
            func_out_params,
            func_call_accord,
        ],
        show_progress=False,
    ).then(
        **enable_chat_input_args
    )
    recall_btn.click(**disable_chat_input_args).then(
        recall_message,
        inputs=state,
        outputs=[state, context_chatbot, raw_context_json],
        show_progress=False,
    ).then(**enable_chat_input_args)
    clear_btn.click(**disable_chat_input_args).then(
        reset_conversation,
        inputs=state,
        outputs=[
            state,
            context_chatbot,
            input_text,
            raw_context_json,
            func_name,
            func_in_params,
            func_out_params,
        ],
    ).then(**enable_chat_input_args)

    chosen_func_names.select(
        try_update_candidates,
        inputs=[
            state,
            chosen_func_names,
            custom_func_code,
            custom_func_desc,
        ],
        outputs=[
            state,
            chosen_func_names,
        ],
        show_progress=False,
    )

    custom_func_code.change(
        remove_old_custom_function,
        inputs=[
            state,
            chosen_func_names,
        ],
        outputs=[
            state,
            chosen_func_names,
        ],
        show_progress=False,
    )
    custom_func_desc.change(
        remove_old_custom_function,
        inputs=[
            state,
            chosen_func_names,
        ],
        outputs=[
            state,
            chosen_func_names,
        ],
        show_progress=False,
    )
    update_func_desc_btn.click(
        update_custom_func_desc,
        inputs=[
            custom_func_code,
            custom_func_desc,
        ],
        outputs=custom_func_desc,
        show_progress=False,
    )

    call_func_btn.click(
        lambda: gr.update(interactive=False),
        outputs=call_func_btn,
        show_progress=False,
        queue=False,
    ).then(
        call_function,
        inputs=[
            state,
            chosen_func_names,
            func_name,
            func_in_params,
        ],
        outputs=func_out_params,
        show_progress=False,
    ).then(
        lambda: gr.update(interactive=True),
        outputs=call_func_btn,
        show_progress=False,
        queue=False,
    )
    send_res_btn.click(**disable_chat_input_args).then(
        generate_response_for_function,
        inputs=[
            state,
            chosen_func_names,
            auth_state,
            ernie_model,
            func_name,
            func_out_params,
            top_p,
            temperature,
        ],
        outputs=[
            state,
            context_chatbot,
            input_text,
            raw_context_json,
            func_name,
            func_in_params,
            func_out_params,
        ],
        show_progress=False,
    ).then(**enable_chat_input_args)
    reset_func_btn.click(
        lambda: (None, None, None),
        outputs=[
            func_name,
            func_in_params,
            func_out_params,
        ],
    )


def create_function_tab(function):
    with gr.Tab(label=function.name):
        with gr.Column():
            gr.Code(label="定义", value=get_source_code(function.func), language="python", interactive=False)
            JSONCode(label="描述", value=to_pretty_json(function.desc, from_json=False), interactive=False)


def make_state_updater(key, *, strip=False):
    def _update_state(state, val):
        if strip:
            val = val.strip()
        state[key] = val
        return state

    return _update_state


def update_api_type(auth_state, api_type):
    auth_state["api_type"] = api_type
    if api_type == "qianfan":
        return auth_state, gr.update(visible=True), gr.update(visible=True)
    elif api_type == "aistudio":
        return auth_state, gr.update(visible=False), gr.update(visible=False)


def generate_response_for_function(
    state,
    candidates,
    auth_config,
    ernie_model,
    func_name,
    func_res,
    top_p,
    temperature,
):
    if text_is_empty(func_name):
        raise gr.Error("函数名称不能为空")
    if func_res is None:
        raise gr.Error("无法获取函数响应参数，请调用函数")
    message = {"role": "function", "name": func_name, "content": to_compact_json(func_res, from_json=True)}
    yield from generate_response(
        state=state,
        candidates=candidates,
        auth_config=auth_config,
        ernie_model=ernie_model,
        message=message,
        top_p=top_p,
        temperature=temperature,
    )


def generate_response_for_text(
    state,
    candidates,
    auth_config,
    ernie_model,
    content,
    top_p,
    temperature,
):
    if text_is_empty(content):
        raise gr.Error("消息内容不能为空")
    content = content.strip().replace("<br>", "\n")
    message = {"role": "user", "content": content}
    yield from generate_response(
        state=state,
        candidates=candidates,
        auth_config=auth_config,
        ernie_model=ernie_model,
        message=message,
        top_p=top_p,
        temperature=temperature,
    )


def recall_message(state):
    context = state["context"]
    if len(context) < 2:
        raise gr.Error("请至少进行一轮对话")
    context = context[:-2]
    history = extract_history(context)
    state["context"] = context
    return state, history, context[-MAX_CONTEXT_LINES_TO_SHOW:]


def regenerate_response(
    state,
    candidates,
    auth_config,
    ernie_model,
    top_p,
    temperature,
):
    context = state["context"]
    if len(context) < 2:
        raise gr.Error("请至少进行一轮对话")
    context.pop()
    user_message = context.pop()
    yield from generate_response(
        state=state,
        candidates=candidates,
        auth_config=auth_config,
        ernie_model=ernie_model,
        message=user_message,
        top_p=top_p,
        temperature=temperature,
    )


def reset_conversation(state):
    state["context"].clear()
    return state, None, None, None, None, None, None


def generate_response(
    state,
    candidates,
    auth_config,
    ernie_model,
    message,
    top_p,
    temperature,
):
    context = copy.copy(state["context"])
    context.append(message)
    name2function = state["name2function"]
    functions = [name2function[name].desc for name in candidates]
    params = {
        "model": ernie_model,
        "messages": context,
        "top_p": top_p,
        "temperature": temperature,
        "functions": functions,
    }

    try:
        resp_stream = create_chat_completion(
            _config_={k: v for k, v in auth_config.items() if v},
            **params,
            stream=True,
        )
    except eb.errors.TokenUpdateFailedError as e:
        raise gr.Error("鉴权参数无效，请重新填写") from e
    except eb.errors.EBError as e:
        raise gr.Error(f"请求失败。错误信息如下：{str(e)}") from e

    context.append({"role": "assistant", "content": None})
    history = None
    for resp in resp_stream:
        if hasattr(resp, "function_call"):
            function_call = resp.function_call
            func_name = function_call["name"]
            try:
                func_args = to_pretty_json(function_call["arguments"], from_json=True)
            except gr.Error:
                # This is most likely because the model is returning incorrectly
                # formatted JSON. In this case we use the raw string.
                func_args = function_call["arguments"]
            context[-1]["function_call"] = function_call
            assert history is None
            history = extract_history(context)
            state["context"] = context
            yield (
                state,
                history,
                None,
                context[-MAX_CONTEXT_LINES_TO_SHOW:],
                func_name,
                func_args,
                None,
                gr.update(open=True),
            )
            break
        else:
            if history is None:
                history = extract_history(context)
            if context[-1]["content"] is None:
                context[-1]["content"] = ""
            old_content = context[-1]["content"]
            delta = resp.result
            context[-1]["content"] += delta
            for content in stream_output_smoother(
                old_content,
                delta,
            ):
                history[-1][1] = content
                yield (
                    state,
                    history,
                    None,
                    *replicate_gradio_update(5),
                )
            assert history[-1][1] == context[-1]["content"]
    else:
        state["context"] = context
        yield (
            state,
            history,
            None,
            context[-MAX_CONTEXT_LINES_TO_SHOW:],
            *replicate_gradio_update(4),
        )


def extract_history(context):
    history = []
    for round_idx in range(0, len(context), 2):
        user_message = context[round_idx]
        pair = []
        if user_message["role"] == "function":
            function_call_result = to_pretty_json(user_message["content"], from_json=True)
            pair.append(
                f"**【函数调用】** 我调用了函数`{user_message['name']}`，函数的返回结果如下：\n\n```\n{function_call_result}\n```"
            )
        elif user_message["role"] == "user":
            pair.append(user_message["content"])
        else:
            raise gr.Error("消息中的`role`不正确")

        try:
            assistant_message = context[round_idx + 1]
        except IndexError:
            pair.append(None)
        else:
            if "function_call" in assistant_message:
                function_call = assistant_message["function_call"]
                function_call_response = to_pretty_json(function_call["arguments"], from_json=True)
                pair.append(
                    (
                        f"**【函数调用】** {function_call['thoughts']}\n\n"
                        f"我建议调用函数`{function_call['name']}`，传入如下参数：\n\n```\n{function_call_response}\n```"
                    )
                )
            else:
                pair.append(assistant_message["content"])

        assert len(pair) == 2
        history.append(pair)
    return history


def stream_output_smoother(old_content, delta, *, delay=0.03):
    content = old_content
    yield content
    for char in delta:
        content += char
        yield content
        time.sleep(delay)


def try_update_candidates(state, candidates, custom_func_code, custom_func_desc_str):
    if CUSTOM_FUNC_NAME in candidates:
        try:
            custom_function = make_custom_function(custom_func_code, custom_func_desc_str)
        except Exception as e:
            handle_exception(e, f"自定义函数的定义或描述中存在错误，无法将其添加为候选函数。错误信息如下：{str(e)}", raise_=False)
            candidates.remove(CUSTOM_FUNC_NAME)
        else:
            state["name2function"][CUSTOM_FUNC_NAME] = custom_function
    return state, candidates


def remove_old_custom_function(state, candidates):
    state["name2function"].pop(CUSTOM_FUNC_NAME, None)
    if CUSTOM_FUNC_NAME in candidates:
        candidates.remove(CUSTOM_FUNC_NAME)
    return state, candidates


def update_custom_func_desc(custom_func_code, custom_func_desc_str):
    try:
        func = code_to_function(custom_func_code, CUSTOM_FUNC_NAME)
        sig = inspect.signature(func)
        custom_func_desc = json_to_obj(custom_func_desc_str)
        new_params_desc = get_custom_func_desc_template()["parameters"]
        for param in sig.parameters.values():
            name = param.name
            if name in custom_func_desc["parameters"]["properties"]:
                param_desc = custom_func_desc["parameters"]["properties"][name]
            else:
                param_desc = {}
            if param.kind in (param.POSITIONAL_ONLY, param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise gr.Error("函数中不可包含positional-only、var-positional或var-keyword参数")
            if param.default is not param.empty:
                param_desc["default"] = param.default
            else:
                if "default" in param_desc:
                    del param_desc["default"]
            if param.kind == param.POSITIONAL_OR_KEYWORD and param.default is param.empty:
                if "required" not in new_params_desc:
                    new_params_desc["required"] = []
                new_params_desc["required"].append(name)
            new_params_desc["properties"][name] = param_desc
        custom_func_desc["parameters"] = new_params_desc
    except Exception as e:
        raise gr.Error(f"更新函数描述失败，错误信息如下：{str(e)}") from e
    else:
        return to_pretty_json(custom_func_desc)


def make_custom_function(code, desc_str):
    func = code_to_function(code, CUSTOM_FUNC_NAME)
    if func.__name__ != CUSTOM_FUNC_NAME:
        raise gr.Error(f"在自定义函数的定义中，必须将函数名称设置为{repr(CUSTOM_FUNC_NAME)}")
    desc = json_to_obj(desc_str)
    if desc["name"] != CUSTOM_FUNC_NAME:
        raise gr.Error(f"在自定义函数的描述中，必须将函数名称设置为{repr(CUSTOM_FUNC_NAME)}")
    return make_function(func, desc, name=CUSTOM_FUNC_NAME)


def call_function(state, candidates, func_name, func_args):
    name2function = state["name2function"]
    if text_is_empty(func_name):
        raise gr.Error("函数名称不能为空")
    if func_name not in name2function:
        raise gr.Error(f"函数`{func_name}`不存在")
    if func_name not in candidates:
        raise gr.Error(f"函数`{func_name}`不是候选函数")
    func = name2function[func_name].func
    if text_is_empty(func_args):
        func_args = "{}"
    func_args = json_to_obj(func_args)
    if not isinstance(func_args, dict):
        raise gr.Error(f"无法将{reprlib.repr(func_args)}解析为字典")
    try:
        res = func(**func_args)
    except Exception as e:
        raise gr.Error(f"函数{func_name}调用失败，错误信息如下：{str(e)}") from e
    return to_pretty_json(res, from_json=False)


JSONCode = functools.partial(gr.Code, language="json")

Function = collections.namedtuple("Function", ["name", "func", "desc"])


def get_custom_func_def_template():
    indent = 2
    return f"def {CUSTOM_FUNC_NAME}():\n{' '*indent}# Write your code here"


def get_custom_func_desc_template():
    return {
        "name": CUSTOM_FUNC_NAME,
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {},
        },
        "responses": {
            "type": "object",
            "properties": {},
        },
    }


def get_predefined_functions():
    functions = []

    def magic_op(a, b):
        return {"result": a * b + (a - b)}

    magic_op_desc = {
        "name": "magic_op",
        "description": "计算输入数字经过“魔法运算”得到的结果",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                },
                "b": {
                    "type": "integer",
                },
            },
            "required": [
                "a",
                "b",
            ],
        },
        "responses": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "integer",
                    "description": "“魔法运算”结果",
                },
            },
        },
    }

    functions.append(make_function(magic_op, magic_op_desc))

    def get_contact_info(name, field=None):
        info_dict = {
            "李小明": {
                "age": 31,
                "email": "lxm@bidu.com",
            },
            "王刚": {
                "age": 28,
                "email": "wg123@bidu.com",
            },
            "张一一": {
                "age": 26,
                "email": "z11@bidu.com",
            },
        }
        info = info_dict[name]
        if field is not None:
            return {"name": name, field: info[field]}
        else:
            return {"name": name, **info}

    get_contact_info_desc = {
        "name": "get_contact_info",
        "description": "获取联系人的个人信息",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "联系人姓名",
                },
                "field": {
                    "type": "string",
                    "description": "想要获取的字段名称，如果不指定则返回所有字段",
                    "enum": [
                        "age",
                        "email",
                    ],
                },
            },
            "required": [
                "name",
            ],
        },
        "responses": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "姓名",
                },
                "age": {
                    "type": "integer",
                    "description": "年龄",
                    "minimum": 0,
                },
                "email": {
                    "type": "string",
                    "description": "电子邮箱地址",
                    "format": "email",
                },
            },
            "required": [
                "name",
            ],
        },
    }

    functions.append(make_function(get_contact_info, get_contact_info_desc))

    return functions


def code_to_function(code, func_name):
    code = compile(code, "<string>", "exec")
    co_names = code.co_names
    if len(co_names) != 1 or co_names[0] != func_name:
        raise gr.Error(f"只允许定义一个函数，函数名为{repr(func_name)}")
    locals_ = {}
    exec(code, {"__builtins__": {}}, locals_)
    func = locals_[func_name]
    return func


def get_source_code(func):
    code = inspect.getsource(func)
    return textwrap.dedent(code)


def make_function(func, desc, *, name=None):
    if name is None:
        name = func.__name__
    return Function(name=name, func=func, desc=desc)


def create_chat_completion(*args, **kwargs):
    response = eb.ChatCompletion.create(*args, **kwargs)
    return response


def json_to_obj(str_):
    try:
        return json.loads(str_)
    except (TypeError, json.JSONDecodeError) as e:
        raise gr.Error(f"无法以JSON格式解码{reprlib.repr(str_)}") from e


def obj_to_json(obj, **kwargs):
    try:
        return json.dumps(obj, ensure_ascii=False, **kwargs)
    except TypeError as e:
        raise gr.Error(f"无法将{reprlib.repr(obj)}编码为JSON") from e


def to_compact_json(obj, *, from_json=False):
    if from_json:
        obj = json_to_obj(obj)
    return obj_to_json(obj, separators=(",", ":"))


def to_pretty_json(obj, *, from_json=False):
    if from_json:
        obj = json_to_obj(obj)
    return obj_to_json(obj, sort_keys=False, indent=2)


def handle_exception(exception, message, *, raise_=False):
    if raise_:
        raise gr.Error(message)
    else:
        traceback.print_exception(type(exception), exception, exception.__traceback__)
        gr.Warning(message)


def replicate_gradio_update(count, /, **kwargs):
    return tuple(gr.update(**kwargs) for _ in range(count))


def text_is_empty(text):
    return text is None or text.strip() == ""


if __name__ == "__main__":
    args = parse_setup_args()
    create_ui_and_launch(args)
