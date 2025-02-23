from typing import List, Dict, Any
import tiktoken
import os
from copy import deepcopy

from .Base import BaseStrategy
from models.Base import BaseModel
from datasets.Dataset import Dataset
from results.Results import Results


class DirectStrategy(BaseStrategy):
    def run_single_pass(self, data_row) -> str:
        """处理单个数据行"""
        try:
            # 处理不同类型的输入数据
            if isinstance(data_row, dict):
                prompt = data_row.get("prompt", "")
            elif isinstance(data_row, list):
                prompt = str(data_row)
            else:
                prompt = str(data_row)
                
            # 构造标准格式的输入
            processed_input = [{
                "role": "user",
                "content": prompt
            }]
            
            response = self.gpt_chat(processed_input=processed_input)
            return response if response else ""
            
        except Exception as e:
            print(f"Direct策略处理错误: {str(e)}")
            return ""
