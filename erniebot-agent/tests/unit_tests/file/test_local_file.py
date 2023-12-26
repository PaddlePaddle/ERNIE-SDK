import pathlib
import tempfile

import erniebot_agent.file.protocol as protocol
from erniebot_agent.file.local_file import LocalFile, create_local_file_from_path


def test_create_local_file_from_path():
    with tempfile.TemporaryDirectory() as td:
        file_path = pathlib.Path(td) / "temp_file"
        file_path.touch()
        file_purpose = "assistants"

        file = create_local_file_from_path(file_path, file_purpose, {})

        assert isinstance(file, LocalFile)
        assert protocol.is_local_file_id(file.id)
