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


class RemoteToolError(Exception):
    def __init__(self, message: str, stage: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.stage = stage

    def __str__(self):
        if not self.stage:
            return self.message

        return f"An error occured in stage <{self.stage}>. The error message is {self.message}"


class FileError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class ObjectClosedError(Exception):
    pass
