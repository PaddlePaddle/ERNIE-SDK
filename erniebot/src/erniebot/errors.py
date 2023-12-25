# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Mapping, Optional

__all__ = [
    "ArgumentNotFoundError",
    "InvalidArgumentError",
    "TokenUpdateFailedError",
    "UnsupportedAPITypeError",
    "HTTPRequestError",
    "ConnectionError",
    "TimeoutError",
    "APIError",
    "BadRequestError",
    "InvalidTokenError",
    "RateLimitError",
    "RequestLimitError",
    "TokenExpiredError",
    "TryAgain",
]


class EBError(Exception):
    """Base exception class for the erniebot library."""


class ArgumentNotFoundError(EBError):
    """An argument was not found."""

    def __init__(self, argument: str) -> None:
        super().__init__(f"Argument `{argument}` not found")


class ConfigItemNotFoundError(EBError):
    """A configuration item was not found."""


class InvalidArgumentError(EBError):
    """An argument is invalid."""


class TokenUpdateFailedError(EBError):
    """The security token could not be updated."""


class UnsupportedAPITypeError(EBError):
    """An unsupported API type was used."""


class HTTPRequestError(EBError):
    """An HTTP request failed."""

    def __init__(
        self,
        message: Optional[str] = None,
        rcode: Optional[int] = None,
        rbody: Optional[str] = None,
        rheaders: Optional[Mapping[str, Any]] = None,
    ) -> None:
        message = self._construct_full_message(message, rcode=rcode, rbody=rbody, rheaders=rheaders)
        super().__init__(message)

    def _construct_full_message(
        self,
        msg: Optional[str],
        rcode: Optional[int],
        rbody: Optional[str],
        rheaders: Optional[Mapping[str, Any]],
    ) -> str:
        parts = []
        msg = msg or ""
        if rcode is not None:
            parts.append(f"code: {rcode}")
        if rbody is not None:
            parts.append(f"body: {repr(rbody)}")
        if rheaders is not None:
            parts.append(f"headers: {rheaders}")
        full_msg = f"{msg}"
        if len(parts) > 0:
            full_msg += " \nResponse:"
            for part in parts:
                full_msg += f"\n  {part}"
        return full_msg


class ConnectionError(HTTPRequestError):
    """Failed to connect to the server."""


class TimeoutError(HTTPRequestError):
    """The operation timed out."""


class APIError(HTTPRequestError):
    """The API responded with an error code."""

    def __init__(
        self,
        message: Optional[str] = None,
        rcode: Optional[int] = None,
        rbody: Optional[str] = None,
        rheaders: Optional[Mapping[str, Any]] = None,
        ecode: Optional[int] = None,
    ) -> None:
        super().__init__(message=message, rcode=rcode, rbody=rbody, rheaders=rheaders)
        self.ecode = ecode


class BadRequestError(APIError):
    """The request was malformed or missing some required parameters."""


class InvalidTokenError(APIError):
    """The security token is invalid."""


class RateLimitError(APIError):
    """The rate limit has been hit."""


class RequestLimitError(APIError):
    """The maximum number of API requests has been exceeded."""


class TokenExpiredError(APIError):
    """The security token has expired."""


class TryAgain(APIError):
    """The API prompted the caller to try again later."""
