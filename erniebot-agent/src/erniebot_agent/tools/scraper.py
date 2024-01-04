from __future__ import annotations

from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any
from langchain.document_loaders import PyMuPDFLoader
from langchain.retrievers import ArxivRetriever
from functools import partial
import requests
from bs4 import BeautifulSoup

from typing import List, Type
from erniebot_agent.tools.base import Tool, ToolParameterView
from pydantic import Field


class ScraperToolInputItemView(ToolParameterView):
    url: str = Field(description="http 有效链接")


class ScraperToolInputView(ToolParameterView):
    urls: List[ScraperToolInputItemView] = Field(description="http 有效链接列表")


class ScraperToolOutputView(ToolParameterView):
    result: List[str] = Field(description="每个 URL 链接中有效的内容")


class ScraperTool(Tool):
    """
    Scraper class to extract the content from the links
    """
    description: str = "能够从 HTTP URL 链接中提取文本内容"
    input_type: Type[ToolParameterView] = ScraperToolInputView
    output_type: Type[ToolParameterView] = ScraperToolOutputView

    def __init__(self, user_agent):
        """
        Initialize the Scraper class.
        Args:
            urls:
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent
        })

    async def __call__(self, urls: List[str]) -> Any:
        urls = [item["url"] for item in urls]
        partial_extract = partial(self.extract_data_from_link, session=self.session)
        with ThreadPoolExecutor(max_workers=20) as executor:
            contents = executor.map(partial_extract, urls)
        res = [content for content in contents if content['raw_content'] is not None]
        return {"result": res}

    def extract_data_from_link(self, link, session):
        """
        Extracts the data from the link
        """
        content = ""
        try:
            if link.endswith(".pdf"):
                content = self.scrape_pdf_with_pymupdf(link)
            elif "arxiv.org" in link:
                doc_num = link.split("/")[-1]
                content = self.scrape_pdf_with_arxiv(doc_num)
            elif link:
                content = self.scrape_text_with_bs(link, session)

            if len(content) < 100:
                return {'url': link, 'raw_content': None}
            return {'url': link, 'raw_content': content}
        except Exception:
            return {'url': link, 'raw_content': None}

    def scrape_text_with_bs(self, link, session):
        response = session.get(link, timeout=4)
        soup = BeautifulSoup(response.content, 'lxml', from_encoding=response.encoding)

        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        raw_content = self.get_content_from_url(soup)
        lines = (line.strip() for line in raw_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        content = "\n".join(chunk for chunk in chunks if chunk)
        return content

    def scrape_pdf_with_pymupdf(self, url) -> str:
        """Scrape a pdf with pymupdf

        Args:
            url (str): The url of the pdf to scrape

        Returns:
            str: The text scraped from the pdf
        """
        loader = PyMuPDFLoader(url)
        doc = loader.load()
        return str(doc)

    def scrape_pdf_with_arxiv(self, query) -> str:
        """Scrape a pdf with arxiv
        default document length of 70000 about ~15 pages or None for no limit

        Args:
            query (str): The query to search for

        Returns:
            str: The text scraped from the pdf
        """
        retriever = ArxivRetriever(load_max_docs=2, doc_content_chars_max=None)
        docs = retriever.get_relevant_documents(query=query)
        return docs[0].page_content

    def get_content_from_url(self, soup):
        """Get the text from the soup

        Args:
            soup (BeautifulSoup): The soup to get the text from

        Returns:
            str: The text from the soup
        """
        text = ""
        tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5']
        for element in soup.find_all(tags):  # Find all the <p> elements
            text += element.text + "\n"
        return text
