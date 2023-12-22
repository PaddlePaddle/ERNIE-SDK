import unittest
from unittest.mock import MagicMock, patch

from erniebot_agent.retrieval.document import Document


class TestDocument(unittest.TestCase):

    def test_document(self):
        document = Document(
            title='This is the title',
            content_se="This is not the answer you are looking for.", 
            meta={"name": "Obi-Wan Kenobi"})
        self.assertEqual(document.title, "This is the title")
        self.assertEqual(document.content_se, "This is not the answer you are looking for.")
        self.assertEqual(document.meta, {'name': 'Obi-Wan Kenobi'})


    def test_dict_convert(self):
        gt_map = {
            'title':"This is the title",
            "content_se":"This is the main content",
            "meta":{
                "name":"doc name"
            }
        }
        document = Document.from_dict(gt_map)
        self.assertEqual(document.id, "5ec3da8d6709c4099ffde0d3bc2b8758")
        self.assertEqual(document.title, "This is the title")
        self.assertEqual(document.content_se, "This is the main content")
        self.assertEqual(document.meta, {'name': 'doc name'})

        data_map = document.to_dict()
        self.assertEqual(gt_map["title"], data_map['title'])
        self.assertEqual(gt_map["content_se"], data_map['content_se'])
        self.assertEqual(gt_map['meta'], data_map['meta'])




