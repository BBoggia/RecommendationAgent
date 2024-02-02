from typing import Any, Dict, List
from langchain.schema import AgentAction
from pydantic import BaseModel

class ContextualData(BaseModel):
    # List of the previously used tools
    tool_history: List[str] = []

    # List of dicts that represent agent actions
    # Each has tool used, input, output, and log
    action_history: List[Dict[str, Any]] = []

    # List of dicts that represent user-agent interactions
    # Each has the user input and agent output
    interaction_history: List[Dict[str, Any]] = []

    def add_agent_action(self, action: AgentAction):
        self.action_history.append({
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log
        })
        self.tool_history.append(action.tool)

    def update_finidhed_agent_action(self, outcome: str):
        self.action_history[-1]["outcome"] = outcome
        self.action_history[-1]["log"] += f"\nOutcome: {outcome}"

    def add_interaction(self, user_input: str, agent_output: str):
        self.interaction_history.append({
            "user_input": user_input,
            "agent_output": agent_output
        })

    def last_tool_used(self):
        return self.tool_history[-1] if self.tool_history else None

    def last_interaction(self):
        return self.interaction_history[-1] if self.interaction_history else None
    
    def serialize_for_gpt4(self):
        # Serialize relevant context into string for llm
        context_str = ''
        for interaction in self.interaction_history:
            context_str += f"Previous Question: {interaction['user_input']}\nPrevious Answer: {interaction['agent_output']}\n"
        # for tool_usage in self.tool_history:
        #     context_str += f"Used tool: {tool_usage['tool']} with input {tool_usage['input']}, got outcome {tool_usage['outcome']}\n"
        return context_str