import base64
import json
from typing import Any, Dict, List, Optional

import requests
from erniebot_agent.utils.exception import APIConnectionError
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
            # '{"errCode": 200101, "errMsg": "project name is exist!"}'
            if result["errCode"] != 0:
                raise APIConnectionError(message=result["errMsg"], error_code=result["errCode"])
            return result
        else:
            raise APIConnectionError(
                message=f"request error: {res.text}", error_code=f"status code: {res.status_code}"
            )

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
                raise APIConnectionError(message=result["errMsg"], error_code=result["errCode"])
            return res.json()
        else:
            raise APIConnectionError(
                message=f"request error: {res.text}", error_code=f"status code: {res.status_code}"
            )

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
                raise APIConnectionError(message=result["errMsg"], error_code=result["errCode"])
            return res.json()
        else:
            raise APIConnectionError(
                message=f"request error: {res.text}", error_code=f"status code: {res.status_code}"
            )

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
                raise APIConnectionError(message=result["errMsg"], error_code=result["errCode"])
            list_data = []
            for item in result["hits"]:
                content = item["_source"]["doc"]
                content = base64.b64decode(content).decode("utf-8")
                json_data = json.loads(content)
                list_data.append(json_data)
            return list_data
        else:
            raise APIConnectionError(
                message=f"request error: {res.text}", error_code=f"status code: {res.status_code}"
            )

    def indexing(self, list_data: List[Document], batch_size: int = 10):
        if type(list_data[0]) == Document:
            list_dict = [item.to_dict() for item in list_data]
        return self.add_documents(list_dict, batch_size=batch_size)

    def get_document_by_id(self, doc_id):
        json_data = {"projectId": self.projectId, "followIndexFlag": True, "dataBody": [doc_id]}
        res = requests.post(f"{self.baseUrl}/baizhong/data-api/v2/flush/get", json=json_data)
        if res.status_code == 200:
            result = res.json()
            if result["errCode"] != 0:
                raise APIConnectionError(message=result["errMsg"], error_code=result["errCode"])
            return result
        else:
            raise APIConnectionError(
                message=f"request error: {res.text}", error_code=f"status code: {res.status_code}"
            )

    def delete_document_by_id(self, doc_id):
        json_data = {"projectId": self.projectId, "followIndexFlag": True, "dataBody": [doc_id]}
        res = requests.post(f"{self.baseUrl}/baizhong/data-api/v2/flush/delete", json=json_data)
        if res.status_code == 200:
            result = res.json()
            if result["errCode"] != 0:
                raise APIConnectionError(message=result["errMsg"], error_code=result["errCode"])
            return result
        else:
            raise APIConnectionError(
                message=f"request error: {res.text}", error_code=f"status code: {res.status_code}"
            )

    def add_documents(self, list_data, batch_size=10):
        for i in tqdm(range(0, len(list_data), batch_size)):
            batch_data = list_data[i : i + batch_size]
            json_data = {"projectId": self.projectId, "followIndexFlag": True, "dataBody": batch_data}
            res = requests.post(f"{self.baseUrl}/baizhong/data-api/v2/flush/add", json=json_data)
            if res.status_code == 200:
                result = res.json()
                if result["errCode"] != 0:
                    raise APIConnectionError(message=result["errMsg"], error_code=result["errCode"])
            else:
                # TODO(wugaosheng): retry 3 times
                raise APIConnectionError(
                    message=f"request error: {res.text}", error_code=f"status code: {res.status_code}"
                )
        return {"errCode": 0, "errMsg": "indexing success!"}

    def delete_all_documents(self, project_id: int):
        # Currently delete all documents means delete project
        self.delete_project(project_id)

    def delete_project(self, project_id: int):
        json_data = {"projectId": project_id}
        res = requests.post(f"{self.baseUrl}/baizhong/web-api/v2/project/delete", json=json_data)
        if res.status_code == 200:
            result = res.json()
            if result["errCode"] != 0:
                raise APIConnectionError(message=result["errMsg"], error_code=result["errCode"])
            return res.json()
        else:
            raise APIConnectionError(
                message=f"request error: {res.text}", error_code=f"status code: {res.status_code}"
            )
