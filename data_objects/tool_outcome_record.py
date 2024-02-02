from datetime import datetime
from typing import Any, Dict
from pydantic import BaseModel

class ToolOutcomeRecord(BaseModel):
    tool_name: str
    timestamp: datetime
    request_args: Dict[str, Any]
    outcome_content: str
    outcome_metadata: Dict[str, Any]  # Data like retrieved document id, etc...