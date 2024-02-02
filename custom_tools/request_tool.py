import re
import requests
from bs4 import BeautifulSoup
from langchain.tools.base import BaseTool
from langchain.utilities import TextRequestsWrapper 

class RequestTool(BaseTool):
    name: str
    description: str
    requests: TextRequestsWrapper

    def __init__(self):
        super(RequestTool, self).__init__(name="Requests", description="Useful for when you need to access a website that does not have an API. Takes a url as input. (Example: 'https://www.google.com')", requests = TextRequestsWrapper(headers=None, aiosession=None))
                                                     # Useful for when you to make a request to a specified URL given to you. Takes a URL as input. (Example: 'https://www.google.com')

    def _run(self, input: str, *args, **kwargs) -> str:
        try:
            response = self.parse_html(self.requests.get(input, **kwargs))
            # forces a request exception to test error handling
            # raise requests.exceptions.ConnectTimeout
            return response
        except requests.exceptions.ConnectionError:
            return "The website you requested is currently unavailable. Please try again later."
        except requests.exceptions.MissingSchema:
            return "The website you requested is invalid. Make sure you are using the correct format. (Example: https://www.google.com)"
        except requests.exceptions.InvalidSchema:
            return "The website you requested is invalid. Make sure you are using the correct format. (Example: https://www.google.com)"
        except requests.exceptions as e:
            print("\n" + "An error occured while trying to access the website you requested. Response from server: " + e.__str__() + "\n")
            return "An error occured while trying to access the website you requested. Response from server: " + e.__str__()
        
    def parse_html(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        text = re.sub('\s{3,}', '\n\n', soup.get_text().strip())
        return text