from models.Gemini import Gemini
from models.OpenAI import OpenAIModel
from models.TencentCloud import TencentCloudModel
from models.Ollama import OllamaModel

class ModelFactory:
    @staticmethod
    def get_model_class(provider: str):
        """获取模型类"""
        provider_map = {
            "OpenAI": OpenAIModel,
            "Gemini": Gemini,
            "TencentCloud": TencentCloudModel,
            "ollama": OllamaModel
        }
        
        if provider not in provider_map:
            raise ValueError(f"不支持的模型提供商: {provider}")
            
        return provider_map[provider]
