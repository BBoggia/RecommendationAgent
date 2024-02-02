from typing import Any, Dict
from langchain.tools.base import BaseTool
from langchain.tools.human import HumanInputRun

class HumanInputTool(BaseTool):
    name: str
    description: str
    human_tool: HumanInputRun

    def __init__(self):
        super(HumanInputTool, self).__init__(name="Human Input", description="Useful for when you need to get information from a human. Takes a question as input. (Example: 'What is your name?')", human_tool = HumanInputRun(input_func=self.human_tool_input))

    def human_tool_input(self) -> str:
        print("Insert your input below. When you're done, type 'done' on a new line and press enter or press CTRL+D.")
        lines = []
        while True:
            try:
                line = input()
                if line == "done":
                    break
                lines.append(line)
            except EOFError:
                break
        return "\n".join(lines)

    def _run(self, input: str, *args, **kwargs) -> str:
        return self.human_tool.run(input)