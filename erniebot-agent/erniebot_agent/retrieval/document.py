import hashlib
import json
from typing import Any, Dict, Optional, Union

from pydantic import BaseConfig
from pydantic.dataclasses import dataclass

BaseConfig.arbitrary_types_allowed = True


@dataclass
class Document:
    id: Optional[str]
    title: str
    content_se: str
    meta: Dict[str, Any]

    def __init__(
        self,
        content_se: str,
        title: str,
        id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.content_se = content_se
        self.title = title
        self.id = id or self._get_id()
        self.meta = meta or {}

    @classmethod
    def _get_id(cls, content_se=None) -> str:
        md5_bytes = content_se.encode(encoding="UTF-8")
        md5_string = hashlib.md5(md5_bytes).hexdigest()
        return md5_string

    def to_dict(self, field_map: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Convert Document to dict. An optional field_map can be supplied to
        change the names of the keys in the resulting dict.
        This way you can work with standardized Document objects in erniebot-agent,
        but adjust the format that they are serialized / stored in other places
        (e.g. elasticsearch)
        Example:

        ```python
            doc = Document(content="some text", content_type="text")
            doc.to_dict(field_map={"custom_content_field": "content"})

            # Returns {"custom_content_field": "some text"}
        ```

        :param field_map: Dict with keys being the custom target keys and values
        being the standard Document attributes
        :return: dict with content of the Document
        """
        if not field_map:
            field_map = {}

        inv_field_map = {v: k for k, v in field_map.items()}
        _doc: Dict[str, str] = {}
        for k, v in self.__dict__.items():
            # Exclude internal fields (Pydantic, ...) fields from the conversion process
            if k.startswith("__"):
                continue
            k = k if k not in inv_field_map else inv_field_map[k]
            _doc[k] = v
        return _doc

    @classmethod
    def from_dict(cls, dict: Dict[str, Any], field_map: Optional[Dict[str, Any]] = None):
        """
        Create Document from dict. An optional `field_map` parameter
        can be supplied to adjust for custom names of the keys in the
        input dict. This way you can work with standardized Document
        objects in erniebot-agent, but adjust the format that
        they are serialized / stored in other places (e.g. elasticsearch).

        Example:

        ```python
            my_dict = {"custom_content_field": "some text", "content_type": "text"}
            Document.from_dict(my_dict, field_map={"custom_content_field": "content"})
        ```

        :param field_map: Dict with keys being the custom target keys and values
        being the standard Document attributes
        :return: A Document object
        """
        if not field_map:
            field_map = {}

        _doc = dict.copy()
        init_args = ["content_se", "meta", "id", "title"]
        if "meta" not in _doc.keys():
            _doc["meta"] = {}
        if "id" not in _doc.keys():
            _doc["id"] = cls._get_id(_doc["content_se"])
        # copy additional fields into "meta"
        for k, v in _doc.items():
            # Exclude internal fields (Pydantic, ...) fields from the conversion process
            if k.startswith("__"):
                continue
            if k not in init_args and k not in field_map:
                _doc["meta"][k] = v
        # remove additional fields from top level
        _new_doc = {}
        for k, v in _doc.items():
            if k in init_args:
                _new_doc[k] = v
            elif k in field_map:
                k = field_map[k]
                _new_doc[k] = v
        return cls(**_new_doc)

    @classmethod
    def from_json(cls, data: Union[str, Dict[str, Any]], field_map: Optional[Dict[str, Any]] = None):
        if not field_map:
            field_map = {}
        if isinstance(data, str):
            dict_data = json.loads(data)
        else:
            dict_data = data
        return cls.from_dict(dict_data, field_map=field_map)
