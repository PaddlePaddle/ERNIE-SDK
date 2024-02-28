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

import asyncio
import weakref
from typing import Any, NoReturn, Optional, final

import asyncio_atexit  # type: ignore[import-untyped]
from typing_extensions import Self

from erniebot_agent.file.file_manager import FileManager
from erniebot_agent.file.remote_file import AIStudioFileClient
from erniebot_agent.utils import config_from_environ as C

_registry: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


@final
class GlobalFileManagerHandler(object):
    """Singleton handler for managing the global FileManager instance.

    This class provides a singleton instance for managing the global FileManager
    and allows for its configuration and retrieval.


    Methods:
        get: Retrieves the global FileManager instance.
        configure: Configures the global FileManager
                   at the beginning of event loop.
        set: Sets the global FileManager explicitly.

    """

    _file_manager: Optional[FileManager]

    def __new__(cls) -> Self:
        loop = asyncio.get_running_loop()
        if loop not in _registry:
            handler = super().__new__(cls)
            handler._file_manager = None
            _registry[loop] = handler
        return _registry[loop]

    def get(self) -> FileManager:
        """
        Retrieve the global FileManager instance.

        This method returns the existing global FileManager instance,
        creating one if it doesn't exist.


        Returns:
            FileManager: The global FileManager instance.

        """
        if self._file_manager is None:
            self._file_manager = self._create_default_file_manager(
                access_token=None,
                save_dir=None,
                enable_remote_file=False,
            )
        return self._file_manager

    def configure(
        self,
        *,
        access_token: Optional[str] = None,
        save_dir: Optional[str] = None,
        enable_remote_file: bool = False,
        **opts: Any,
    ) -> None:
        """
        Configure the global FileManager.

        This method configures the global FileManager with the provided parameters
        at the beginning of event loop.
        If the global FileManager is already set, it raises an error.

        Args:
            access_token (Optional[str]): The access token for remote file client.
            save_dir (Optional[str]): The directory for saving local files.
            enable_remote_file (bool): Whether to enable remote file.
            **opts (Any): Additional options for FileManager.

        Returns:
            None

        Raises:
            RuntimeError: If the global FileManager is already set.

        """
        if self._file_manager is not None:
            self._raise_file_manager_already_set_error()
        self._file_manager = self._create_default_file_manager(
            access_token=access_token,
            save_dir=save_dir,
            enable_remote_file=enable_remote_file,
            **opts,
        )

    def set(self, file_manager: FileManager) -> None:
        """
        Set the global FileManager explicitly.

        This method sets the global FileManager instance explicitly.
        If the global FileManager is already set, it raises an error.

        Args:
            file_manager (FileManager): The FileManager instance to set as global.

        Returns:
            None

        Raises:
            RuntimeError: If the global FileManager is already set.
        """
        if self._file_manager is not None:
            self._raise_file_manager_already_set_error()
        self._file_manager = file_manager

    def _create_default_file_manager(
        self,
        access_token: Optional[str],
        save_dir: Optional[str],
        enable_remote_file: bool,
        **opts: Any,
    ) -> FileManager:
        """Create the default FileManager instance."""

        async def _close_file_manager():
            await file_manager.close()

        if access_token is None:
            access_token = C.get_global_access_token()
        if save_dir is None:
            save_dir = C.get_global_save_dir()
        remote_file_client = None
        if enable_remote_file:
            if access_token is None:
                raise RuntimeError("An access token must be provided to enable remote file management.")
            remote_file_client = AIStudioFileClient(access_token=access_token)
        file_manager = FileManager(remote_file_client, save_dir, **opts)
        asyncio_atexit.register(_close_file_manager)
        return file_manager

    def _raise_file_manager_already_set_error(self) -> NoReturn:
        raise RuntimeError(
            "The global file manager can only be set once."
            " The setup can be done explicitly by calling `configure` or `set`,"
            " or implicitly by calling `get`."
        )
