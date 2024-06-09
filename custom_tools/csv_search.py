import pandas as pd
from io import IOBase
from langchain.agents.agent import AgentExecutor
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from typing import List, Union, Optional, Any
from custom_tools.human_input_tool import HumanInputTool
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.output_parsers import RetryOutputParser
from langchain.chat_models.base import BaseChatModel
from langchain_experimental.agents.agent_toolkits.pandas.prompt import SUFFIX_NO_DF, SUFFIX_WITH_DF, SUFFIX_WITH_MULTI_DF

PREFIX = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df` and containes Home Depot's product catalogue.
To accomplish your task, think about the steps you would take to find the information in the dataframe. If your first approach doesn't work, try another one. If you are stuck, you can ask the user for help.
You should use the tools below to answer the question posed of you:"""


class CSVTool(BaseTool):
    """Tool for working with CSV files using pandas."""
    path_list: Union[str, List[str]]
    llm: BaseChatModel
    csv_agent: AgentExecutor
    pandas_kwargs: Optional[dict]
    dataframes: Union[pd.DataFrame, List[pd.DataFrame]] = None
    
    
    def __init__(self, llm, path_list: Union[str, List[str]], config: Any, human_input_tool: HumanInputTool = None, pandas_kwargs: Optional[dict] = {}):
        """Initialize the CSVTool with path(s) to a CSV file(s) and optional pandas arguments.

        Args:
            path: One or more paths to CSV files or file-like objects.
            pandas_kwargs: Optional dictionary of arguments to pass to pandas.read_csv.
        """
        super(CSVTool, self).__init__(
                name="CSV Data Tool", 
                description="Useful for when you need to access data from a CSV file. Takes a plain text question as an input. Make sure inputs are thorough and robust.", 
                path_list = path_list, 
                llm = llm, 
                pandas_kwargs = pandas_kwargs,
                csv_agent = create_csv_agent(llm=llm, path=path_list, extra_tools=[human_input_tool], prefix=PREFIX, verbose = True)) # config['tool_settings']['verbose']))
        
        self.dataframes = self._load_csvs()

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
        if isinstance(self.path_list, (str, IOBase)):
            df = self._load_csv(self.path_list)
        elif isinstance(self.path_list, list):
            df = [self._load_csv(item) for item in self.path_list]
        else:
            raise ValueError(f"Expected str, list, or file-like object, got {type(self.path_list)}")
        return df or []

    def _run(self, input: str, *args, **kwargs) -> str:
        """Run an action using the DataFrame(s) stored in the tool.
        
        Args:
            input: The action to perform on the DataFrame(s).
            kwargs: Additional keyword arguments to pass to the action call.
        
        Returns:
            The result of the action call.
        """
        print("INPUT: ", input)

        print("BEFORE INVOKING CSV AGENT")
        result = self.csv_agent.invoke({"input": input})
        print("AFTER INVOKING CSV AGENT\n", result)
            
        return result
        
        # Migrate from using seperate agent to using same agent as other tools