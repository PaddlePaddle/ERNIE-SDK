import logging
from typing import Dict, List, Optional

import requests

_logger = logging.getLogger(__name__)


class CustomSearch:
    def __init__(self, base_url, outId, key, access_token=None):
        self._base_url = base_url
        self.outId = outId
        self.key = key
        self.access_token = access_token
        if self.access_token is None:
            self.access_token = self._get_ticket()

    def _get_ticket(
        self,
    ):
        res = requests.post(
            f"{self._base_url}/api/account/getticket?outId={self.outId}&key={self.key}",
        )
        result = res.json()
        return result["Data"]

    def _get_authorization_headers(self, access_token: Optional[str]) -> Dict:
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

    def search(self, searchKeywords: str, identifier: str = "U", top_k: int = 10, **kwargs) -> List[Dict]:
        data = {
            "pageSize": top_k,
            "searchKeywords": searchKeywords,
            "identifier": identifier,
        }
        data.update(kwargs)
        res = requests.post(
            f"{self._base_url}/api/search/getarticlesearchresult",
            headers=self._get_authorization_headers(access_token=self.access_token),
            params=data,
        )
        if res.status_code == 200:
            result = res.json()
            return result
        else:
            raise Exception(f"Error: {res.text}")
