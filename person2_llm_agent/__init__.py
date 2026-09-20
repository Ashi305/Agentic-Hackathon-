"""
Person 2: Agent & LLM Engine
Responsible for prompt engineering, dynamic few-shot override injection,
RAG knowledge retrieval, and multi-step agent orchestration.
"""
from .prompts import PromptFactory, build_extraction_prompt, build_classification_prompt
from .few_shot_manager import FewShotManager, DynamicFewShotManager
from .rag_engine import RAGEngine, SafetyRAGEngine
from .tools import extract_json, validate_extraction, validate_classification
from .agent_orchestrator import SafetyReportAgent

IncidentPrecursorAgent = SafetyReportAgent
SafetyToolbox = dict

__all__ = [
    "PromptFactory",
    "DynamicFewShotManager",
    "FewShotManager",
    "SafetyRAGEngine",
    "RAGEngine",
    "SafetyToolbox",
    "SafetyReportAgent",
    "IncidentPrecursorAgent",
    "extract_json",
    "validate_extraction",
    "validate_classification",
]
