import json

from langchain.output_parsers.json import parse_json_markdown


class JsonUtil:
    def parse_json(self, json_str, start_indicator: str = "{", end_indicator: str = "}"):
        if start_indicator == "{":
            response = parse_json_markdown(json_str)
        else:
            start_idx = json_str.index(start_indicator)
            end_idx = json_str.rindex(end_indicator)
            corrected_data = json_str[start_idx : end_idx + 1]
            response = json.loads(corrected_data)
        return response
