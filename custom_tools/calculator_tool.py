from langchain.chains.llm_math.base import LLMMathChain
from langchain_core.tools import BaseTool
    
    
class CalculatorTool(BaseTool):
    name: str
    description: str
    math: LLMMathChain

    def __init__(self, llm, config):
        super(CalculatorTool, self).__init__(name="Calculator", description="Useful for when you need to do math.", math = LLMMathChain.from_llm(llm, verbose = config['tool_settings']['verbose']))

    def _run(self, input: str, *args, **kwargs) -> str:
        return self.math.run(input)
    
class RoundingTool(BaseTool):
    name: str
    description: str

    def __init__(self):
        super(RoundingTool, self).__init__(name="Rounding", description="Useful for when you need to round a number to the nearest n decimal places. Takes a number and a decimal place as input. (Example: '1.2345, 2' -> 1.23)")

    def _run(self, input: str, *args, **kwargs) -> str:
        '''Takes a number and a decimal place as input and returns the number rounded to the nearest n decimal places.\n
        Example: "1.2345, 2" -> 1.23
        '''
        num, decimal = input.split(', ')
        if decimal == 0:
            return round(float(num))
        else:
            return round(float(num), int(decimal))