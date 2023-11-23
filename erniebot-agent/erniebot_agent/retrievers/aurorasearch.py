import base64
import json
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

from .document import Document


class AuroraSearch:
    def __init__(self, baseUrl: str, projectName: str, remark: str, max_seq_length: int = 512) -> None:
        self.baseUrl = baseUrl
        self.projectName = projectName
        self.remark = remark
        self.index = self.create_project()
        print(self.index)
        # self.projectId = self.index['result']['projectId']
        self.projectId = 274
        self.max_seq_length = max_seq_length
        self.create_schema()

    def create_project(self):
        json_data = {
            "projectName": self.projectName,
            "remark": self.remark,
        }
        res = requests.post(f"{self.baseUrl}/baizhong/web-api/v2/project/add", json=json_data)
        if res.status_code == 200:
            result = res.json()
            # '{"errCode": 200101, "errMsg": "project name is exist!"}'
            if result["errCode"] == 200101:
                return result
            return result

    def create_schema(self, max_seq_length: int = 512):
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
            return res.json()

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
            list_data = []
            for item in result["hits"]:
                content = item["_source"]["doc"]
                content = base64.b64decode(content).decode("utf-8")
                json_data = json.loads(content)
                list_data.append(json_data)
            return list_data
        else:
            raise Exception(f"request error: {res.status_code}")

    def indexing(self, list_data: List[Document]):
        if type(list_data[0]) == Document:
            list_dict = [item.to_dict() for item in list_data]
        return self.add_documents(list_dict)

    def get_document_by_id(self, doc_id):
        pass

    def add_documents(self, list_data, batch_size=10):
        for i in tqdm(range(0, len(list_data), batch_size)):
            batch_data = list_data[i : i + batch_size]
            json_data = {"projectId": self.projectId, "followIndexFlag": True, "dataBody": batch_data}
            res = requests.post(f"{self.baseUrl}/baizhong/data-api/v2/flush/add", json=json_data)

            if res.status_code == 200:
                msg = res.json()
                print(msg)
        return {"message": "success"}

    def delete_all_documents(self, project_id: int):
        json_data = {"projectId": project_id}
        res = requests.post(f"{self.baseUrl}/baizhong/baizhong/web-api/v2/project/delete", json=json_data)
        if res.status_code == 200:
            return res.json()

    def delete_project(self, project_id: int):
        json_data = {"projectId": project_id}
        res = requests.post(f"{self.baseUrl}/baizhong/data-api/v2/flush/add", json=json_data)
        if res.status_code == 200:
            return res.json()
