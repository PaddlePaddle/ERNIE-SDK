import openai
import os

openai.api_key = "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"

stream = True
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Hello!"
    }],
    stream=stream)

if not stream:
    print(completion)
else:
    collected_messages = list()
    for chunk in completion:
        chunk_message = chunk['choices'][0]['delta']
        collected_messages.append(chunk_message)
    print(collected_messages)
