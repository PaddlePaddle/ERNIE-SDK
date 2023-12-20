import json
import logging
import os
from typing import Any, ClassVar, Dict, Optional

import requests

from erniebot_agent.utils.exception import BaizhongError

logger = logging.getLogger(__name__)


class BaizhongSearch:
    _AISTUDIO_BASE_URL: ClassVar[str] = "https://aistudio.baidu.com"

    def __init__(
        self,
        access_token: str,
        knowledge_base_name: Optional[str] = None,
        knowledge_base_id: Optional[int] = None,
    ) -> None:
        self.base_url = os.getenv("AISTUDIO_BASE_URL", self._AISTUDIO_BASE_URL)
        self.access_token = access_token
        if knowledge_base_id is not None:
            logger.info(f"Loading existing project with `knowledge_base_id={knowledge_base_id}`")
            self.knowledge_base_id = knowledge_base_id
        elif knowledge_base_name is not None:
            self.knowledge_base_id = self.create_knowledge_base(knowledge_base_name)

        else:
            raise BaizhongError("You must provide either a `knowledge_base_name` or a `knowledge_base_id`.")

    def create_knowledge_base(self, knowledge_base_name: str):
        json_data = {"knowledgeBaseName": knowledge_base_name}
        res = requests.post(
            f"{self.base_url}/llm/knowledge/create",
            json=json_data,
            headers=self._get_authorization_headers(access_token=self.access_token),
        )
        if res.status_code == 200:
            result = res.json()
            if result["errorCode"] != 0:
                raise BaizhongError(message=result["errorMsg"], error_code=result["errorCode"])
            return result["result"]["knowledgeBaseId"]
        else:
            raise BaizhongError(message=f"request error: {res.text}", error_code=res.status_code)

    def _get_authorization_headers(self, access_token: Optional[str]) -> dict:
        headers = {"Content-Type": "application/json"}
        if access_token is None:
            logger.warning("access_token is NOT provided, this may cause 403 HTTP error..")
        else:
            headers["Authorization"] = f"token {access_token}"
        return headers

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
        json_data = {"knowledgeBaseId": self.knowledge_base_id, "query": query, "topK": top_k}
        if filters is not None:
            filter_terms = [{"term": item} for item in filters]
            filterConditions = {"filterConditions": {"bool": {"filter": filter_terms}}}
            json_data.update(filterConditions)
        res = requests.post(
            f"{self.base_url}/llm/knowledge/search",
            json=json_data,
            headers=self._get_authorization_headers(access_token=self.access_token),
        )
        if res.status_code == 200:
            result = res.json()
            if result["errorCode"] != 0:
                raise BaizhongError(message=result["errorMsg"], error_code=result["errorCode"])
            list_data = []
            for item in result["result"]:
                doc = json.loads(item["source"]["doc"])
                list_data.append(
                    {
                        "id": item["fileId"],
                        "content": doc["content_se"],
                        "title": item["source"]["title"],
                        "score": item["score"],
                    }
                )
            return list_data
        else:
            raise BaizhongError(message=f"request error: {res.text}", error_code=res.status_code)
