import base64
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import requests
from erniebot_agent.utils.exception import BaizhongError
from tqdm import tqdm

from erniebot.utils.logging import logger

from .document import Document


class BaizhongSearch:
    def __init__(
        self, baseUrl: str, projectName: str, remark: str, projectId: int = -1, max_seq_length: int = 512
    ) -> None:
        self.baseUrl = baseUrl
        self.projectName = projectName
        self.remark = remark
        self.max_seq_length = max_seq_length
        if projectId == -1:
            logger.info("Creating new project and schema")
            self.index = self.create_project()
            logger.info("Project creation succeeded")
            self.projectId = self.index["result"]["projectId"]
            self.create_schema()
            logger.info("Schema creation succeeded")

        else:
            logger.info("Loading existing project with `project_id={projectId}`")
            self.projectId = projectId

    def create_project(self):
        json_data = {
            "projectName": self.projectName,
            "remark": self.remark,
        }
        res = requests.post(f"{self.baseUrl}/baizhong/web-api/v2/project/add", json=json_data)
        if res.status_code == 200:
            result = res.json()
            if result["errCode"] != 0:
                raise BaizhongError(message=result["errMsg"], error_code=result["errCode"])
            return result
        else:
            raise BaizhongError(message=f"request error: {res.text}", error_code=res.status_code)

    def create_schema(self):
        json_data = {
            "projectId": self.projectId,
            "schemaJson": {
                "paraSize": self.max_seq_length,
                "dataSegmentationMod": "neisou",
                "storeType": "ElasticSearch",
                "properties": {
                    "title": {"type": "text", "shortindex": True},
                    "content_se": {"type": "text", "longindex": True},
                },
            },
        }
        res = requests.post(f"{self.baseUrl}/baizhong/web-api/v2/project-schema/create", json=json_data)
        if res.status_code == 200:
            result = res.json()
            if result["errCode"] != 0:
                raise BaizhongError(message=result["errMsg"], error_code=result["errCode"])
            return res.json()
        else:
            raise BaizhongError(message=f"request error: {res.text}", error_code=res.status_code)

    def update_schema(
        self,
    ):
        json_data = {
            "projectId": self.projectId,
            "schemaJson": {
                "paraSize": self.max_seq_length,
                "dataSegmentationMod": "neisou",
                "storeType": "ElasticSearch",
                "properties": {
                    "title": {"type": "text", "shortindex": True},
                    "content_se": {"type": "text", "longindex": True},
                },
            },
        }
        res = requests.post(f"{self.baseUrl}/baizhong/web-api/v2/project-schema/update", json=json_data)
        status_code = res.status_code
        if status_code == 200:
            result = res.json()
            if result["errCode"] != 0:
                raise BaizhongError(message=result["errMsg"], error_code=result["errCode"])
            return result
        else:
            raise BaizhongError(message=f"request error: {res.text}", error_code=res.status_code)

    def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None):
        json_data = {
            "query": query,
            "projectId": self.projectId,
            "size": top_k,
        }
        if filters is not None:
            filterConditions = {"filterConditions": {"bool": {"filter": {"match": filters}}}}
            json_data.update(filterConditions)
        res = requests.post(f"{self.baseUrl}/baizhong/common-search/v2/search", json=json_data)
        if res.status_code == 200:
            result = res.json()
            if result["errCode"] != 0:
                raise BaizhongError(message=result["errMsg"], error_code=result["errCode"])
            list_data = []
            for item in result["hits"]:
                content = item["_source"]["doc"]
                content = base64.b64decode(content).decode("utf-8")
                json_data = json.loads(content)
                list_data.append(json_data)
            return list_data
        else:
            raise BaizhongError(message=f"request error: {res.text}", error_code=res.status_code)

    def indexing(self, list_data: List[Document], batch_size: int = 10, thread_count: int = 2):
        if type(list_data[0]) == Document:
            list_dicts = [item.to_dict() for item in list_data]
        all_data = []
        for i in tqdm(range(0, len(list_dicts), batch_size)):
            batch_data = list_data[i : i + batch_size]
            all_data.append(batch_data)
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            res = executor.map(self.add_documents, all_data)
        return res

    def get_document_by_id(self, doc_id):
        json_data = {"projectId": self.projectId, "followIndexFlag": True, "dataBody": [doc_id]}
        res = requests.post(f"{self.baseUrl}/baizhong/data-api/v2/flush/get", json=json_data)
        if res.status_code == 200:
            result = res.json()
            if result["errCode"] != 0:
                raise BaizhongError(message=result["errMsg"], error_code=result["errCode"])
            return result
        else:
            raise BaizhongError(message=f"request error: {res.text}", error_code=res.status_code)

    def delete_documents(
        self,
        ids: Optional[List[str]] = None,
    ):
        json_data = {"projectId": self.projectId, "followIndexFlag": True}
        if ids:
            json_data["dataBody"] = ids
        else:
            # TODO: delete all documents
            raise NotImplementedError
        res = requests.post(f"{self.baseUrl}/baizhong/data-api/v2/flush/delete", json=json_data)
        if res.status_code == 200:
            result = res.json()
            if result["errCode"] != 0:
                raise BaizhongError(message=result["errMsg"], error_code=result["errCode"])
            return result
        else:
            raise BaizhongError(message=f"request error: {res.text}", error_code=res.status_code)

    def add_documents(self, documents: List[Dict[str, Any]]):
        json_data = {"projectId": self.projectId, "followIndexFlag": True, "dataBody": documents}
        res = requests.post(f"{self.baseUrl}/baizhong/data-api/v2/flush/add", json=json_data)
        if res.status_code == 200:
            result = res.json()
            if result["errCode"] != 0:
                raise BaizhongError(message=result["errMsg"], error_code=result["errCode"])
            return result
        else:
            # TODO(wugaosheng): retry 3 times
            raise BaizhongError(message=f"request error: {res.text}", error_code=res.status_code)

    @classmethod
    def delete_project(cls, project_id: int):
        json_data = {"projectId": project_id}
        res = requests.post(f"{cls.baseUrl}/baizhong/web-api/v2/project/delete", json=json_data)
        if res.status_code == 200:
            result = res.json()
            if result["errCode"] != 0:
                raise BaizhongError(message=result["errMsg"], error_code=result["errCode"])
            return res.json()
        else:
            raise BaizhongError(message=f"request error: {res.text}", error_code=res.status_code)
