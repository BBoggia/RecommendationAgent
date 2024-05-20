import pprint
from uuid import UUID
from typing import Any, Dict, List, Optional
from managers import RichContextMemory
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackHandler

class MemoryCallbackHandler(BaseCallbackHandler):
    def __init__(self, memory: RichContextMemory):
        super().__init__()
        self.memory = memory

    def on_agent_action(self, action: AgentAction, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        print("\n\nAgent action: ")

        print(f"    tool: {action.tool}")
        print(f"    tool_input: {action.tool_input}")

        log_parts = action.log.split('\n')
        print(f"    log: {log_parts[0]}")
        for part in log_parts[1:]:
            print(f"         {part}")
        self.memory.add_agent_action(action)  # Uncommented this line to add agent action to memory
        super().on_agent_action(action, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any,) -> Any:
        print("\n\nTool start: ", serialized, "\n\nKWARGS: ", kwargs, "\nINPUT: ", input_str, "\nMetadata: ", metadata, "\n")
        return super().on_tool_start(serialized, input_str, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)    
    
    def on_tool_end(self, output: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any, ) -> Any:
        print("\nTool end: ", output, "\nKWARGS: ", kwargs, "\n")
        self.memory.update_finished_agent_action(output)  # Added this line to update finished agent action in memory
        return super().on_tool_end(output, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID = None, tags: List[str] = None, metadata: Dict[str, Any] = None, **kwargs: Any) -> Any:
        print("\n\nChain start: \nKWARGS: ", kwargs, "\nInputs: ", inputs, "\nMetadata: ", metadata, "\n")
        return super().on_chain_start(serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)
    
    def on_text(self, text: str, *, run_id: UUID, parent_run_id: UUID = None, **kwargs: Any) -> Any:
        print("\nText", "KWARGS: ", kwargs, "\n")
        return super().on_text(text, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID = None, **kwargs: Any) -> Any:
        print("\nChain end:\nOutput: ", outputs, "KWARGS: ", kwargs, "\n")
        return super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
