from __future__ import annotations

import base64
import unittest

import pytest

from erniebot_agent.tools.remote_tool import RemoteToolError, check_base64_string
from erniebot_agent.tools.utils import is_base64_string


class TestUtils(unittest.TestCase):
    def test_is_base64_string(self):
        base64_string = base64.b64encode(b"sss").decode()

        self.assertTrue(is_base64_string(base64_string))

        self.assertFalse(is_base64_string("sss"))

    def test_check_base64_string(self):
        base64_string = base64.b64encode(b"sss").decode()

        with pytest.raises(RemoteToolError):
            check_base64_string({"file": base64_string})

        with pytest.raises(RemoteToolError):
            check_base64_string({"file": [base64_string]})

        with pytest.raises(RemoteToolError):
            check_base64_string({"file": {"file": [base64_string]}})
