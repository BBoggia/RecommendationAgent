from typing import Any, Dict
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import BaseTool

class GoogleSearchTool(BaseTool):
    name: str
    description: str
    config: Dict[str, Any]
    google_search: GoogleSearchAPIWrapper

    def __init__(self, config):
        super(GoogleSearchTool, self).__init__(name="Google Search", description="Useful for when you need to search google for the most recent and up to date time sensitive information. (Example: 'Who do the Cleveland Browns play this week?')", config = config, handle_parsing_errors = True, google_search = GoogleSearchAPIWrapper())

    def _run(self, input: str, *args, **kwargs) -> str:
        return self.google_search.run(input)