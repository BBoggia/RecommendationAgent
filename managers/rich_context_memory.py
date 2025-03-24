from typing import Any, List, Tuple
from langchain.memory import ConversationBufferMemory
from pydantic import Field
from data_objects import ContextualData
from langchain_core.agents import AgentAction


class RichContextMemory(ConversationBufferMemory):
    contextual_data: ContextualData = ContextualData()

    def __init__(self, memory_key: str, return_messages: bool = True, json_safe: bool = False, max_history: int = 10):
        super().__init__(memory_key=memory_key, return_messages=return_messages, json_safe=json_safe, max_history=max_history)

    def remember(self, **kwargs):
        # Can directly manipulate the memory object here (dont forget)
        if "user_input" in kwargs and "agent_output" in kwargs:
            self.contextual_data.add_interaction(kwargs["user_input"], kwargs["agent_output"])

    def add_agent_action(self, action: AgentAction):
        self.contextual_data.add_agent_action(action)

    def update_finished_agent_action(self, outcome: str):
        self.contextual_data.update_finidhed_agent_action(outcome)

    def get_context_data(self):
        return self.contextual_data
    
    def get_history_serialized(self):
        # Utilize serialize_for_gpt4 method from contextual_data to get the serialized string
        return self.contextual_data.serialize_for_gpt4()
    
    def finish_last_tool(self, outcome: str):
        self.contextual_data.finish_last_tool(outcome)
    
    def update(self, steps: List[Tuple[str, Any]]) -> None:
        """
        Update the in-memory store of the agent's interactions and tool usages.
        Args:
            steps: A list of tuples containing the interactions and responses.
        """
        for action, outcome in steps:
            if isinstance(action, AgentAction):
                # If action is AgentAction update tool interaction history
                self.contextual_data.add_agent_action(action, outcome)
            else:
                # If action is final interaction add to interaction history
                self.contextual_data.add_interaction(action, outcome)
                
