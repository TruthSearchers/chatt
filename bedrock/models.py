# models.py

from typing import Dict, List, Union
from config import config
from langchain_aws import ChatBedrockConverse

class ChatModel:
    def __init__(self, model_name: str, model_kwargs: Dict, bedrock_runtime=None):
        """
        Initialize the ChatModel with specific model configuration.
        
        Args:
            model_name: Name of the model to use
            model_kwargs: Model configuration parameters
            bedrock_runtime: Optional boto3 bedrock-runtime client
        """
        self.model_config = config["models"][model_name]
        self.model_id = self.model_config["model_id"]
        self.model_kwargs = model_kwargs
        
        # Base parameters for the model
        model_params = {
            "model": self.model_id,
            "max_tokens": self.model_kwargs["max_tokens"],
            "temperature": self.model_kwargs["temperature"],
            "top_p": self.model_kwargs["top_p"]
        }
        
        # Add client if provided
        if bedrock_runtime:
            model_params["client"] = bedrock_runtime
            
        # Add top_k for non-mistral models
        if "mistral" not in self.model_id:
            model_params["additional_model_request_fields"] = {
                "top_k": self.model_kwargs["top_k"]
            }
            
        self.llm = ChatBedrockConverse(**model_params)