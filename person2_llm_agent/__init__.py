"""
Person 2: Agent & LLM Engine
Responsible for prompt engineering, dynamic few-shot override injection,
RAG OSHA knowledge retrieval, decorated tool calling, and multi-step agent orchestration.
"""
from .prompts import PromptFactory
from .few_shot_manager import DynamicFewShotManager
from .rag_engine import SafetyRAGEngine
from .tools import SafetyToolbox
from .agent_orchestrator import IncidentPrecursorAgent

__all__ = [
    "PromptFactory",
    "DynamicFewShotManager",
    "SafetyRAGEngine",
    "SafetyToolbox",
    "IncidentPrecursorAgent",
]
