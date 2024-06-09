import os
import glob
import logging
from typing import Iterable
from langchain_community.document_loaders import Docx2txtLoader, TextLoader, PyPDFLoader, JSONLoader
from data_objects import DocumentInfo
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
        

class DocumentManager:
    def __init__(self, config):
        self.config = config
        root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.document_path = os.path.join(root_directory, self.config['document_paths']['data_docs_dir'])
        self.db_directory = os.path.join(root_directory, self.config['document_paths']['database_dir'])
        self.doc_list = self._get_document_list()

    def _get_document_list(self):
        doc_list = []
        for doc in self.config['documents']:
            try:
                doc_list.append(DocumentInfo(doc['file_name'], doc['name'], doc['description']))
            except KeyError as e:
                logging.error(f"Error loading document info: {e}")

        return doc_list

    def _get_local_docs(self) -> list:
        files = glob.glob(os.path.join(self.document_path, "*.txt"))
        files.extend(glob.glob(os.path.join(self.document_path, "*.pdf")))
        files.extend(glob.glob(os.path.join(self.document_path, "*.docx")))
        files.extend(glob.glob(os.path.join(self.document_path, "*.json")))
        
        docs: Iterable[Document] = []
        for file in files:
            loader = self._get_loader_for_file(file)
            docs.extend(loader.load())

        print(f"{len(files)} files have been loaded with a total of {len(docs)} documents")
        return docs
    
    def _get_loader_for_file(self, file):
        if file.endswith('.txt') or file.endswith('.json'):
            return TextLoader(file)
        elif file.endswith('.pdf'):
            return PyPDFLoader(file)
        elif file.endswith('.docx'):
            return Docx2txtLoader(file)
        # elif file.endswith('.json'):
        #     return JSONLoader(file)
        else:
            # raise ValueError(f"No loader found for: {file}")
            print(f"No loader found for: {file}")

    def _split_docs(self, docs, chunk_size: int, overlap: int):
        chunk_size = self.config['vector_store_settings']['chunk_size'] if chunk_size is None else chunk_size
        overlap = self.config['vector_store_settings']['overlap'] if overlap is None else overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        split_docs = text_splitter.split_documents(docs)
        print(f"{len(docs)} documents were split into {len(split_docs)} new documents")
        return split_docs
    
    def _get_db_ready_docs(self, chunk_size = None, overlap = None):
        chunk_size = self.config['vector_store_settings']['chunk_size'] if chunk_size is None else chunk_size
        overlap = self.config['vector_store_settings']['overlap'] if overlap is None else overlap
        docs = self._get_local_docs()
        split_docs = self._split_docs(docs, chunk_size, overlap)
        return split_docs

    def build_vector_store(self, embedding, docs: Iterable[Document] = None, chunk_size: int = None, overlap: int = None):
        if docs is None:
            chunk_size = self.config['vector_store_settings']['chunk_size'] if chunk_size is None else chunk_size
            overlap = self.config['vector_store_settings']['overlap'] if overlap is None else overlap
            docs = self._get_db_ready_docs(chunk_size, overlap)

        if os.path.exists(self.db_directory):
            vector_store = FAISS(embedding_function=embedding, persist_directory=self.db_directory)
        else:
            vector_store = FAISS.from_documents(docs, embedding=embedding, persist_directory=self.db_directory)
        return vector_store