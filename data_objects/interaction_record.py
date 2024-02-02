from datetime import datetime
from typing import Any, Dict
from pydantic import BaseModel

class InteractionRecord(BaseModel):
    interaction_type: str  # "user" for user inputs, "agent" for agent outputs
    timestamp: datetime  # When interaction happened
    content: str  # Interaction content 
    metadata: Dict[str, Any]  # Additional data like email message IDs, URLs, or relevant titles