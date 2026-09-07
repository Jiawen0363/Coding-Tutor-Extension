#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for student level judgment.
"""

from typing import List
from chatarena.message import Message


def format_conversation_history(history_messages: List[Message]) -> str:
    """
    Format conversation history from Message objects to a string.
    
    Args:
        history_messages: List of Message objects
    
    Returns:
        Formatted conversation string, e.g., "tutor: ...\nstudent: ..."
    """
    conversation_lines = []
    for msg in history_messages:
        conversation_lines.append(f"{msg.agent_name}: {msg.content}")
    return "\n".join(conversation_lines)


def call_level_judge_model(vllm_backend, level_eval_prompt: str) -> str:
    """
    Call the Qwen3-4B model to judge student level.
    
    Args:
        vllm_backend: VLLMChat backend instance
        level_eval_prompt: The filled prompt for level evaluation
    
    Returns:
        Model output (should be "low", "medium", or "high")
    """
    # Construct message for the model
    messages = [{"role": "user", "content": level_eval_prompt}]
    
    # Call the model using the backend's _get_response method
    response = vllm_backend._get_response(messages)
    
    # Clean up the response
    response = response.strip().lower()
    
    return response


def parse_level_output(model_output: str) -> str:
    """
    Parse and validate the model output.
    
    Args:
        model_output: Raw model output
    
    Returns:
        Validated level string ("low", "medium", or "high")
    """
    model_output = model_output.strip().lower()
    
    # Check if the output contains one of the expected levels
    if "low" in model_output:
        return "low"
    elif "medium" in model_output or "med" in model_output:
        return "medium"
    elif "high" in model_output:
        return "high"
    else:
        # Default to medium if unable to parse
        print(f"⚠️ Warning: Unable to parse level from output: '{model_output}'. Defaulting to 'medium'.")
        return "medium"

