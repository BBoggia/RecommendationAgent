from langchain.tools.base import BaseTool
from langchain.chains import RetrievalQA


class DocumentRetrievalTool(BaseTool):
    name: str
    description: str
    retrieval: RetrievalQA

    def __init__(self, config, llm, vector_store):
        super(DocumentRetrievalTool, self).__init__(name="Document Retrieval", description="Useful for when you need to retrieve a document from the document store.", retrieval = RetrievalQA.from_chain_type(llm = llm, retriever = vector_store.as_retriever(), verbose = config['tool_settings']['verbose']))

    def _run(self, input: str, *args, **kwargs) -> str:
        return self.retrieval.run(input)
