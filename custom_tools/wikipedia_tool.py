from typing import Any, Dict
from langchain.tools.base import BaseTool
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.tools.wikipedia.tool import WikipediaQueryRun
    
class WikipediaTool(BaseTool):
    name: str
    description: str
    config: Dict[str, Any]
    wikipedia: WikipediaQueryRun

    def __init__(self, config):
        super(WikipediaTool, self).__init__(name="Wikipedia", description="Useful for when you need to search for information on specific person, place, subject, or thing. Not for time sensitive or recent information. Fallback to the search tool if the Wikipedia tool fails.", config = config, wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()))

    def _run(self, input: str, *args, **kwargs) -> str:
        return self.wikipedia.run(input)