from typing import Any, Dict
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import BaseTool
    
class WikipediaTool(BaseTool):
    name: str
    description: str
    config: Dict[str, Any]
    wikipedia: WikipediaQueryRun

    def __init__(self, config):
        super(WikipediaTool, self).__init__(name="Wikipedia", description="Useful for when you need to search for information on specific person, place, subject, or thing. Not for time sensitive or recent information. Fallback to the search tool if the Wikipedia tool fails.", config = config, handle_parsing_errors = True, wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()))

    def _run(self, input: str, *args, **kwargs) -> str:
        return self.wikipedia.run(input)