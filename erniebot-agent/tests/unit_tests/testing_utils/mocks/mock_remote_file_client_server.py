import contextlib
import re
import uuid

import erniebot_agent.file_io.protocol as protocol
from erniebot_agent.file_io.remote_file import RemoteFile, RemoteFileClient


class FakeRemoteFileProtocol(object):
    _UUID_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    _FILE_ID_PREFIX = r"file-fake-remote-"
    _FILE_ID_PATTERN = _FILE_ID_PREFIX + _UUID_PATTERN

    _followed = False

    @classmethod
    def is_remote_file_id(cls, str_):
        return re.fullmatch(cls._FILE_ID_PATTERN, str_) is not None

    @classmethod
    def extract_remote_file_ids(cls, str_):
        return re.findall(cls._FILE_ID_PATTERN, str_)

    @classmethod
    def generate_remote_file_id(cls):
        return cls._FILE_ID_PREFIX + str(uuid.uuid4())

    get_timestamp = protocol.get_timestamp

    @classmethod
    @contextlib.contextmanager
    def follow(cls, old_protocol=None):
        if not cls._followed:
            names_methods_to_monkey_patch = (
                "is_remote_file_id",
                "extract_remote_file_ids",
                "get_timestamp",
            )

            if old_protocol is None:
                old_protocol = protocol

            _old_methods = {}

            for method_name in names_methods_to_monkey_patch:
                old_method = getattr(old_protocol, method_name)
                new_method = getattr(cls, method_name)
                _old_methods[method_name] = old_method
                setattr(old_protocol, method_name, new_method)

            cls._followed = True

            yield

            for method_name in names_methods_to_monkey_patch:
                old_method = _old_methods[method_name]
                setattr(old_protocol, method_name, old_method)

            cls._followed = False

        else:
            yield


class FakeRemoteFileClient(RemoteFileClient):
    _protocol = FakeRemoteFileProtocol

    def __init__(self, server):
        super().__init__()
        if server.protocol is not self.protocol:
            raise ValueError("Server and client do not share the same protocol.")
        self._server = server

    @property
    def protocol(self):
        return self._protocol

    @property
    def server(self):
        if not self._server.started:
            raise RuntimeError("Server is not running.")
        return self._server

    async def upload_file(self, file_path, file_purpose, file_metadata):
        result = await self.server.upload_file(file_path, file_purpose, file_metadata)
        return self._create_file_obj_from_dict(result)

    async def retrieve_file(self, file_id):
        result = await self.server.retrieve_file(file_id)
        return self._create_file_obj_from_dict(result)

    async def retrieve_file_contents(self, file_id):
        return await self.server.retrieve_file_contents(file_id)

    async def list_files(self):
        result = await self.server.list_files()
        files = []
        for item in result:
            file = self._create_file_obj_from_dict(item)
            files.append(file)
        return files

    async def delete_file(self, file_id) -> None:
        await self.server.delete_file(file_id)

    async def create_temporary_url(self, file_id, expire_after):
        raise RuntimeError("Method not supported")

    def _create_file_obj_from_dict(self, dict_):
        with self._protocol.follow():
            return RemoteFile(
                id=dict_["id"],
                filename=dict_["filename"],
                byte_size=dict_["byte_size"],
                created_at=dict_["created_at"],
                purpose=dict_["purpose"],
                metadata=dict_["metadata"],
                client=self,
            )


class FakeRemoteFileServer(object):
    _protocol = FakeRemoteFileProtocol

    def __init__(self):
        super().__init__()
        self._storage = None

    @property
    def protocol(self):
        return self._protocol

    @property
    def storage(self):
        return self._storage

    @property
    def started(self):
        return self._storage is not None

    async def upload_file(self, file_path, file_purpose, file_metadata):
        id_ = self._protocol.generate_remote_file_id()
        filename = file_path.name
        byte_size = file_path.stat().st_size
        created_at = self._protocol.get_timestamp()
        with file_path.open("rb") as f:
            contents = f.read()
        file = dict(
            id=id_,
            filename=filename,
            byte_size=byte_size,
            created_at=created_at,
            purpose=file_purpose,
            metadata=file_metadata,
            contents=contents,
        )
        self._storage[id_] = file
        return file

    async def retrieve_file(self, file_id):
        try:
            return self._storage[file_id]
        except KeyError as e:
            raise RuntimeError("File not found") from e

    async def retrieve_file_contents(self, file_id):
        try:
            file = self._storage[file_id]
        except KeyError as e:
            raise RuntimeError("File not found") from e
        else:
            return file["contents"]

    async def list_files(self):
        return list(self._storage.values())

    async def delete_file(self, file_id) -> None:
        try:
            return self._storage[file_id]
        except KeyError as e:
            raise RuntimeError("File not found") from e

    @contextlib.contextmanager
    def start(self):
        self._storage = {}
        yield self
        self._storage = None
