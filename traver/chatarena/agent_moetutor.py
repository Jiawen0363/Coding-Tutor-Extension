from typing import List, Tuple, Union
from tenacity import RetryError
import logging
import uuid
import tiktoken

from .backends import IntelligenceBackend
from .message import Message
from .config import AgentConfig, BackendConfig
from .agent import Player

# A special signal sent by the player to indicate that it is not possible to continue the conversation, and it requests to end the conversation.
# It contains a random UUID string to avoid being exploited by any of the players.
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"


class MonitoredTutor(Player):
    """
    Custom tutor agent with student level monitoring.
    Uses a separate Qwen3-4B model to monitor student level before generating responses.
    """
    
    def __init__(self, name: str, role_desc: str, 
                 backend: Union[BackendConfig, IntelligenceBackend],  # Tutor model
                 level_monitor_backend: IntelligenceBackend,           # Qwen3-4B for level monitoring
                 prompt_element: dict,                                 # Task data (namespace, function_name, etc.)
                 adapter_names: dict = None,                           # Adapter names for different levels
                 **kwargs):
        super().__init__(name=name, role_desc=role_desc, backend=backend, **kwargs)
        self.level_monitor_backend = level_monitor_backend
        self.prompt_element = prompt_element
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.level_monitor_history = []  # Store all monitor inputs and outputs
        
        # Adapter names mapping
        if adapter_names is None:
            self.adapter_names = {
                "low": "low_level_tutor",
                "medium": "medium_level_tutor",
                "high": "high_level_tutor"
            }
        else:
            self.adapter_names = adapter_names
        
        # Track current adapter and set initial model
        self.current_adapter = self.adapter_names["medium"]  # Default to medium
        self.backend.model = self.current_adapter  # Set the initial model
    
    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role_desc=self.role_desc,
            backend=self.backend.to_config()
        )
    
    def act(self, observation: List[Message]) -> str:
        """
        Generate a response with student level monitoring.
        """
        try:
            # Import here to avoid circular dependency
            from utils.make_prompt import prompt_student_level_eval
            
            # 1. Format conversation history
            all_messages = []
            for msg in observation:
                all_messages.append(f"{msg.agent_name}: {msg.content}")
            
            if len(all_messages) > 0:  # Skip first turn (tutor speaks first)
                # 2. Fill the level evaluation prompt template
                conversation_str = '\n'.join(all_messages)
                level_eval_prompt = prompt_student_level_eval(
                    self.prompt_element, 
                    self.tokenizer, 
                    conversation_str
                )
                
                # 3. Call Qwen3-4B to judge student level
                query_messages = [{"role": "user", "content": level_eval_prompt}]
                level_output = self.level_monitor_backend._get_response(query_messages)
                
                # 4. Parse level output
                from utils.level_judge_utils import parse_level_output
                parsed_level = parse_level_output(level_output)
                
                # 5. Switch adapter based on level
                new_adapter = self.adapter_names[parsed_level]
                if new_adapter != self.current_adapter:
                    print(f"🔄 Switching adapter: {self.current_adapter} → {new_adapter}")
                    self.current_adapter = new_adapter
                    self.backend.model = new_adapter
                
                # 6. Save monitor data
                turn_number = len(self.level_monitor_history) + 1
                monitor_record = {
                    "turn": turn_number,
                    "monitor_input": level_eval_prompt,
                    "monitor_output": level_output.strip(),
                    "parsed_level": parsed_level,
                    "adapter_used": new_adapter
                }
                self.level_monitor_history.append(monitor_record)
                
                # 7. Print result
                print(f"📊 Turn {turn_number} - Student Level: {parsed_level} (using {new_adapter})")
            
            # 5. Generate tutor response normally
            response = self.backend.query(
                agent_name=self.name, 
                role_desc=self.role_desc,
                history_messages=observation
            )
            
        except RetryError as e:
            logging.warning(f"Agent {self.name} failed to generate a response. "
                          f"Error: {e.last_attempt.exception()}. "
                          f"Sending signal to end the conversation.")
            response = SIGNAL_END_OF_CONVERSATION
        
        return response
    
    def __call__(self, observation: List[Message]) -> str:
        return self.act(observation)
    
    def reset(self):
        self.backend.reset()
        self.level_monitor_history = []
        self.current_adapter = self.adapter_names["medium"]  # Reset to default