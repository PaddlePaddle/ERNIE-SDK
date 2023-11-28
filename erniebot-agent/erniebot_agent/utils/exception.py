from typing import Any, Optional


class BaizhongError(Exception):
    """Exception for issues that occur in a document store"""

    def __init__(self, message: Optional[str] = None, error_code: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def __str__(self):
        if self.error_code:
            docs_message = f" error code is {self.error_code}"
            return "error message is " + self.message + docs_message
        return self.message
