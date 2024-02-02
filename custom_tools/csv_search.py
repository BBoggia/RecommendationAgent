import pandas as pd
from io import IOBase
from langchain.tools.base import BaseTool
from langchain.agents.agent_types import AgentType
from langchain.agents.agent import AgentExecutor
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from typing import List, Union, Optional, Any

from custom_tools.human_input_tool import HumanInputTool


class CSVTool(BaseTool):
    """Tool for working with CSV files using pandas."""
    path_list: Union[str, List[str]]
    csv_agent: AgentExecutor
    
    def __init__(self, llm, path_list: Union[str, List[str]], config: Any, human_input_tool: HumanInputTool = None, pandas_kwargs: Optional[dict] = None):
        """Initialize the CSVTool with path(s) to a CSV file(s) and optional pandas arguments.

        Args:
            path: One or more paths to CSV files or file-like objects.
            pandas_kwargs: Optional dictionary of arguments to pass to pandas.read_csv.
        """
        super(CSVTool, self).__init__(name="CSV Data Tool", description="Useful for when you need to access data from a CSV file. Takes a plain text question as an input. Make sure inputs are thorough and robust.", path_list = path_list, csv_agent = create_csv_agent(llm=llm, path=path_list, extra_tools=[human_input_tool], verbose = True)) # config['tool_settings']['verbose']))
        self.path_list = path_list
        print("PATH LIST: ", self.path_list)
        #self.pandas_kwargs = pandas_kwargs or {}
        # self.dataframes = self._load_csvs()

    def _load_csv(self, path: str) -> pd.DataFrame:
        """Load a single CSV file to a pandas DataFrame.

        Args:
            path: Path to the CSV file or file-like object.
        
        Returns:
            A pandas DataFrame created from the CSV file.
        """
        return pd.read_csv(path, **self.pandas_kwargs)
    
    def _load_csvs(self) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Load CSV file(s) into pandas DataFrame(s).

        Returns:
            A DataFrame or list of DataFrames from the CSV file(s).
        """
        if isinstance(self.path, (str, IOBase)):
            df = self._load_csv(self.path)
        elif isinstance(self.path, list):
            df = [self._load_csv(item) for item in self.path]
        else:
            raise ValueError(f"Expected str, list, or file-like object, got {type(self.path)}")
        return df

    def _run(self, input: str, *args, **kwargs) -> str:
        """Run an action using the DataFrame(s) stored in the tool.
        
        Args:
            input: The action to perform on the DataFrame(s).
            kwargs: Additional keyword arguments to pass to the action call.
        
        Returns:
            The result of the action call.
        """
        return self.csv_agent.invoke({"input": input})
        
        # Migrate from using seperate agent to using same agent as other tools