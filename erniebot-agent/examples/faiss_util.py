import argparse
import os

from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import FAISS
from utils import read_data

embeddings = OpenAIEmbeddings(deployment="text-embedding-ada")


def get_args():
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('--faiss_name', default="faiss_index", help="The faiss index")
    parser.add_argument('--summary_name', default="faiss_index", help="The summary text index")
    parser.add_argument('--fulltext_name', default="faiss_index", help="The full text index")
    parser.add_argument('--file_path', default="data/output.jsonl", help="The data output path")
    parser.add_argument('--indexing_type', choices=['summary_fulltext', 'common'],
                        default="common", help="The indexing types")
    args = parser.parse_args()
    # yapf: enable
    return args


class SemanticSearch:
    def __init__(self, faiss_name, file_path=None) -> None:
        self.faiss_name = faiss_name
        self.file_path = file_path
        self.vector_db = self.init_db()

    def init_db(
        self,
    ):
        if os.path.exists(self.faiss_name):
            faiss = FAISS.load_local(self.faiss_name, embeddings)
        else:
            loader = UnstructuredFileLoader(self.file_path)
            documents = loader.load()
            text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size=1500, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)
            faiss = FAISS.from_documents(docs, embeddings)
            faiss.save_local(self.faiss_name)
        return faiss

    def search(self, query, top_k=4):
        return self.vector_db.similarity_search(query, k=top_k)


class RecursiveDocuments:
    def __init__(self, summary_name, fulltext_name, file_path=None) -> None:
        self.summary_name = summary_name
        self.fulltext_name = fulltext_name
        self.file_path = file_path
        self.vector_db = self.init_db()

    def init_db(
        self,
    ):
        if os.path.exists(self.summary_name) and os.path.exists(self.fulltext_name):
            summary_faiss = FAISS.load_local(self.summary_name, embeddings)
            fulltext_faiss = FAISS.load_local(self.fulltext_name, embeddings)
        else:
            list_data = read_data(self.file_path)
            doc_summary = []
            doc_fulltext = []
            text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size=1500, chunk_overlap=0)
            for item in list_data:
                full_texts = Document(page_content=item["content"])

                abstract = Document(page_content=item["abstract"])
                docs = text_splitter.split_documents([full_texts])
                doc_fulltext.extend(docs)

                doc_summary.append(abstract)

            summary_faiss = FAISS.from_documents(doc_summary, embeddings)
            summary_faiss.save_local(self.summary_name)

            fulltext_faiss = FAISS.from_documents(doc_fulltext, embeddings)
            fulltext_faiss.save_local(self.fulltext_name)
        return summary_faiss

    def search(self, query, top_k=4):
        return self.vector_db.similarity_search(query, k=top_k)


if __name__ == "__main__":
    query = "GPT-3是怎么训练得到的？"
    args = get_args()
    if args.indexing_type == "common":
        faiss_search = SemanticSearch(args.faiss_name, args.file_path)
        docs = faiss_search.search(query)
        print(docs)
    else:
        recursive_search = RecursiveDocuments(args.summary_name, args.fulltext_name, args.file_path)
