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

from typing import (Any, Mapping, Optional)

__all__ = [
    'ArgumentNotFoundError', 'InvalidArgumentError', 'MaxRetriesExceededError',
    'TokenUpdateFailedError', 'UnsupportedAPITypeError', 'HTTPRequestError',
    'ConnectionError', 'TimeoutError', 'APIError', 'InvalidParameterError',
    'InvalidTokenError', 'PermissionError', 'RequestLimitError',
    'ServiceUnavailableError', 'TokenExpiredError', 'TryAgain'
]


class EBError(Exception):
    """Base exception class for the erniebot library."""
    pass


class ArgumentNotFoundError(EBError):
    """Exception that's raised when an argument is not found."""
    pass


class InvalidArgumentError(EBError):
    """Exception that's raised when an argument is invalid."""
    pass


class MaxRetriesExceededError(EBError):
    """Exception that's raised when the maximum number of retries is
    exceeded."""
    pass


class TokenUpdateFailedError(EBError):
    """Exception that's raised when the security token cannot be updated."""
    pass


class UnsupportedAPITypeError(EBError):
    """Exception that's raised when an unsupported API type is used."""
    pass


class HTTPRequestError(EBError):
    """Exception that's raised when an HTTP request fails."""

    def __init__(self,
                 message: Optional[str]=None,
                 rcode: Optional[int]=None,
                 rbody: Optional[str]=None,
                 rheaders: Optional[Mapping[str, Any]]=None) -> None:
        """Initialize the instance based on an error message and an HTTP
        response.

        Args:
            message: Description of the error.
            rcode: HTTP response code.
            rbody: HTTP response body.
            rheaders: HTTP response headers.
        """
        message = self._construct_full_message(
            message, rcode=rcode, rbody=rbody, rheaders=rheaders)
        super().__init__(message)

    def _construct_full_message(self,
                                msg: Optional[str],
                                rcode: Optional[int],
                                rbody: Optional[str],
                                rheaders: Optional[Mapping[str, Any]]) -> str:
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
            full_msg += f" \nResponse:"
            for part in parts:
                full_msg += f"\n  {part}"
        return full_msg


class ConnectionError(HTTPRequestError):
    """Exception that's raised when failing to connect to the server."""
    pass


class TimeoutError(HTTPRequestError):
    """Excpetion that's raised when the request times out."""
    pass


class APIError(HTTPRequestError):
    """Exception that's raised when the API responds with an error code."""

    def __init__(self,
                 message: Optional[str]=None,
                 rcode: Optional[int]=None,
                 rbody: Optional[str]=None,
                 rheaders: Optional[Mapping[str, Any]]=None,
                 ecode: Optional[int]=None) -> None:
        super().__init__(
            message=message, rcode=rcode, rbody=rbody, rheaders=rheaders)
        self.ecode = ecode


class InvalidParameterError(APIError):
    """Exception that's raised when at least one of the passed parameters is
    invalid."""
    pass


class InvalidTokenError(APIError):
    """Exception that's raised when the security token is invalid."""
    pass


class PermissionError(APIError):
    """Exception that's raised when an API request is denied due to lack of
    permission."""
    pass


class RequestLimitError(APIError):
    """Exception that's raised when the maximum number of API requests is
    exceeded."""
    pass


class ServiceUnavailableError(APIError):
    """Exception that's raised when the service is unavailable."""
    pass


class TokenExpiredError(APIError):
    """Exception that's raised when the security token is expired."""
    pass


class TryAgain(APIError):
    """Exception that's raised when the API prompts the caller to try again
    later."""
    pass
