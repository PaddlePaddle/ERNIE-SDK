from __future__ import annotations

import base64
import unittest

from erniebot_agent.tools.utils import is_base64_string


class TestUtils(unittest.TestCase):
    def test_is_base64_string(self):
        base64_string = base64.b64encode(b"sss").decode()

        self.assertTrue(is_base64_string(base64_string))

        self.assertFalse(is_base64_string("sss"))
