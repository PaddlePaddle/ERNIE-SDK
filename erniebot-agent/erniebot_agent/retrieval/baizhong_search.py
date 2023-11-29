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
        """
        Create a project using the Baizhong API.

        Returns:
            dict: A dictionary containing information about the created project.

        Raises:
            BaizhongError: If the API request fails, this exception is raised with details about the error.
        """
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
        """
        Create a schema for a project using the Baizhong API.

        Returns:
            dict: A dictionary containing information about the created schema.

        Raises:
            BaizhongError: If the API request fails, this exception is raised with details about the error.
        """
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
        """
        Update the schema for a project using the Baizhong API.

        Returns:
            dict: A dictionary containing information about the updated schema.

        Raises:
            BaizhongError: If the API request fails, this exception is raised with details about the error.
        """
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
        """
        Perform a search using the Baizhong common search API.

        Args:
            query (str): The search query.
            top_k (int, optional): The number of top results to retrieve (default is 10).
            filters (Optional[Dict[str, Any]], optional): Additional filters to apply to the search query
            (default is None).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing search results.

        Raises:
            BaizhongError: If the API request fails, this exception is raised with details about the error.
        """
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

    def add_documents(self, documents: List[Document], batch_size: int = 10, thread_count: int = 1):
        """
        Add a batch of documents to the Baizhong system using multi-threading.

        Args:
            documents (List[Document]): A list of Document objects to be added.
            batch_size (int, optional): The size of each batch of documents (default is 10).
            thread_count (int, optional): The number of threads to use for concurrent document addition
            (default is 1).

        Returns:
            List[Union[None, Exception]]: A list of results from the document addition process.

        Note:
            This function uses multi-threading to improve the efficiency of adding a large number of
            documents.

        """
        if type(documents[0]) == Document:
            list_dicts = [item.to_dict() for item in documents]
        all_data = []
        for i in tqdm(range(0, len(list_dicts), batch_size)):
            batch_data = list_dicts[i : i + batch_size]
            all_data.append(batch_data)
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            res = executor.map(self._add_documents, all_data)
        return list(res)

    def get_document_by_id(self, doc_id):
        """
        Retrieve a document from the Baizhong system by its ID.

        Args:
            doc_id: The ID of the document to retrieve.

        Returns:
            dict: A dictionary containing information about the retrieved document.

        Raises:
            BaizhongError: If the API request fails, this exception is raised with details about the error.
        """
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
        """
        Delete documents from the Baizhong system.

        Args:
            ids (Optional[List[str]], optional): A list of document IDs to delete. If not provided,
            all documents will be deleted.

        Returns:
            dict: A dictionary containing information about the deletion process.

        Raises:
            NotImplementedError: If the deletion of all documents is attempted, this exception is raised
            as it is not yet implemented.
            BaizhongError: If the API request fails, this exception is raised with details about the error.
        """
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

    def _add_documents(self, documents: List[Dict[str, Any]]):
        """
        Internal method to add a batch of documents to the Baizhong system.

        Args:
            documents (List[Dict[str, Any]]): A list of dictionaries representing documents to be added.

        Returns:
            dict: A dictionary containing information about the document addition process.

        Raises:
            BaizhongError: If the API request fails, this exception is raised with details about the error.
        """
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
        """
        Class method to delete a project using the Baizhong API.

        Args:
            project_id (int): The ID of the project to be deleted.

        Returns:
            dict: A dictionary containing information about the deletion process.

        Raises:
            BaizhongError: If the API request fails, this exception is raised with details about the error.
        """
        json_data = {"projectId": project_id}
        res = requests.post(f"{cls.baseUrl}/baizhong/web-api/v2/project/delete", json=json_data)
        if res.status_code == 200:
            result = res.json()
            if result["errCode"] != 0:
                raise BaizhongError(message=result["errMsg"], error_code=result["errCode"])
            return res.json()
        else:
            raise BaizhongError(message=f"request error: {res.text}", error_code=res.status_code)
