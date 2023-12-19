import argparse
import os

import jsonlines
from utils import create_abstract, erniebot_chat, read_data, split_text

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--api_type", default=None, type=str, help="The API Key.")
parser.add_argument("--access_token", default=None, type=str, help="The secret key.")
parser.add_argument("--data_path", default='data/json_data.jsonl', type=str, help="The data path.")
parser.add_argument("--output_path", default='data/finance_abstract', type=str, help="The output path.")
parser.add_argument('--chatbot_type', choices=['erniebot'], default="erniebot",
                    help="The chatbot model types")
args = parser.parse_args()
# yapf: enable


def summarize_text(text: str):
    if not text:
        return "Error: No text to summarize"
    summaries = []

    chunks = list(split_text(text, max_length=4096))
    scroll_ratio = 1 / len(chunks)
    print(scroll_ratio)
    print(f"Summarizing text with total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        messages = [create_abstract(chunk)]
        summary = erniebot_chat(messages, api_type=args.api_type, access_token=args.access_token).rbody[
            "result"
        ]
        print(summary)
        summaries.append(summary)

    # breakpoint()
    combined_summary = "\n".join(summaries)
    combined_summary = combined_summary[:7000]
    messages = [create_abstract(combined_summary)]

    final_summary = erniebot_chat(messages, api_type=args.api_type, access_token=args.access_token)
    print("Final summary length: ", len(final_summary))
    print(final_summary)
    return final_summary


def generate_summary_jsonl():
    os.makedirs(args.output_path, exist_ok=True)
    list_data = read_data(args.data_path)
    for md_file in list_data:
        markdown_text = md_file["content"]
        summary = summarize_text(markdown_text)
        md_file["abstract"] = summary

    output_json = f"{args.output_path}/data.jsonl"
    with jsonlines.open(output_json, "w") as f:
        for item in list_data:
            f.write(item)
    return output_json


if __name__ == "__main__":
    # text summarization
    generate_summary_jsonl()
