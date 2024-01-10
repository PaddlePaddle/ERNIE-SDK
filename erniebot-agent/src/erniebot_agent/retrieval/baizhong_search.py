import json
import logging
import os
from typing import Any, ClassVar, Dict, Optional

import requests

from erniebot_agent.utils import config_from_environ as C
from erniebot_agent.utils.exceptions import BaizhongError

_logger = logging.getLogger(__name__)


class BaizhongSearch:
    """
    A class for interacting with the Baizhong Search API.


    Attributes:
        base_url (str): The base URL for the AIStudio service.
        access_token (str): The access token for authentication.
        knowledge_base_id (int): The ID of the knowledge base being used (if applicable).
    """

    _AISTUDIO_BASE_URL: ClassVar[str] = "https://aistudio.baidu.com"

    def __init__(
        self,
        access_token: Optional[str] = None,
        knowledge_base_name: Optional[str] = None,
        knowledge_base_id: Optional[int] = None,
    ) -> None:
        """
        Initialize a BaizhongSearch object.

        Args:
            access_token (str): The access token for authentication.
            knowledge_base_name (Optional[str]): The name of the knowledge base to use (optional).
            knowledge_base_id (Optional[int]): The ID of an existing knowledge base to use (optional).

        Raises:
            BaizhongError: If neither knowledge_base_name nor knowledge_base_id is provided.

        """
        self._base_url = os.getenv("AISTUDIO_BASE_URL", self._AISTUDIO_BASE_URL)
        self.access_token: Optional[str] = None
        if access_token is not None:
            self.access_token = access_token
        elif C.get_global_access_token() is not None:
            self.access_token = C.get_global_access_token()
        else:
            raise BaizhongError(
                "Please ensure that either an access_token is provided or "
                "the EB_AGENT_ACCESS_TOKEN is set up as an environment variable."
            )
        if knowledge_base_id is not None:
            _logger.info(f"Loading existing project with `knowledge_base_id={knowledge_base_id}`")
            self.knowledge_base_id = knowledge_base_id
        elif knowledge_base_name is not None:
            self.knowledge_base_id = self.create_knowledge_base(knowledge_base_name)
        else:
            raise BaizhongError("You must provide either a `knowledge_base_name` or a `knowledge_base_id`.")

    def create_knowledge_base(self, knowledge_base_name: str):
        """
        Create a JSON payload with the provided knowledge base name.

        Args:
            knowledge_base_name (str): The knowledge base name.

        Returns:
            Dict[str, Any]: A dictionary containing knowledge base results.

        Raises:
            BaizhongError: If the API request fails, this exception is raised with details about the error.
        """
        json_data = {"knowledgeBaseName": knowledge_base_name}
        res = requests.post(
            f"{self._base_url}/llm/knowledge/create",
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
        """
        Initialize a dictionary for HTTP headers with Content-Type set to application/json.

        Args:
            access_token (str): The AIStudio access_token.

        Returns:
            Dict[str, Any]: A dictionary containing HTTP headers information.
        """
        headers = {"Content-Type": "application/json"}
        if access_token is None:
            _logger.warning("access_token is NOT provided, this may cause 403 HTTP error..")
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
            f"{self._base_url}/llm/knowledge/search",
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
