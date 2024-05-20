from custom_tools import WikipediaTool, GoogleSearchTool, RequestTool, CalculatorTool, RoundingTool, DocumentRetrievalTool, HumanInputTool, CSVTool, TextToSpeechTool  # Added TextToSpeechTool
import os
from langchain_core.tools import Tool

class AgentToolkit:

    def __init__(self, llm, config, vector_store = None) -> None:

        self.llm = llm
        self.vector_store = vector_store
        self.config = config

        self.tool_list = self._load_tools()
         
    def use_own_knowledge(self, input):
        return self.llm.generate(input)
        
    def _load_tools(self) -> list[Tool]:
        # human_input_tool = HumanInputTool()
        return [
            GoogleSearchTool(config = self.config),
            WikipediaTool(config = self.config),
            CalculatorTool(self.llm, self.config),
            RoundingTool(),
            # human_input_tool,
            HumanInputTool(),
            CSVTool(llm=self.llm, path_list="/Users/bransonboggia/Desktop/programming_related/kent-research/deep_learning/amazon_data/home_depot_data.csv", human_input_tool = HumanInputTool(), config=self.config),
            TextToSpeechTool()  # Added this line
        ] # + [DocumentRetrievalTool(self.llm, self.vector_store, config = self.config)] if self.vector_store else []
