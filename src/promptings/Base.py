from typing import List
import tiktoken
import os
import copy
import time

from models.Base import BaseModel
from datasets.Dataset import Dataset
from results.Results import Results
from utils.parse import parse_response
from time import perf_counter_ns
from constants.verboseType import *

class BaseStrategy(object):
    def __init__(
        self,
        model: BaseModel,
        data: Dataset,
        language: str,
        pass_at_k: int,
        results: Results,
        verbose: int = VERBOSE_FULL,
    ):
        self.model = model
        self.data = data
        self.pass_at_k = pass_at_k
        self.results = results
        self.language = language
        self.verbose = verbose
        self.run_details = []
    

    def append_run_details(self, run_details: dict):
        for key in run_details.keys():
            if key in self.run_details:
                self.run_details[key] += run_details[key]
            else:
                self.run_details[key] = run_details[key]


    def gpt_chat(
            self, 
            processed_input: List[dict], 
            frequency_penalty=0, 
            presence_penalty=0
        ):
        
        response, run_details = self.model.prompt(
            processed_input=processed_input, 
            frequency_penalty=frequency_penalty, 
            presence_penalty=presence_penalty
        )
        self.append_run_details(run_details)
        
        return response


    def run_single_pass(self, data_row: dict):
        pass

    def run(self, save_details=False):
        """运行代码生成"""
        self.run_details = {}  # 初始化为空字典
        total_tasks = len(self.data)
        success_count = 0
        solved_count = 0
        
        for idx, data_row in enumerate(self.data, 1):
            try:
                # 标准化数据行格式
                if isinstance(data_row, list):
                    task_id = f"task_{idx}"
                    normalized_data = {
                        "task_id": task_id,
                        "prompt": str(data_row)
                    }
                elif isinstance(data_row, dict):
                    task_id = data_row.get("task_id", f"task_{idx}")
                    normalized_data = data_row
                else:
                    task_id = f"task_{idx}"
                    normalized_data = {
                        "task_id": task_id,
                        "prompt": str(data_row)
                    }
                
                response = self.run_single_pass(normalized_data)
                
                # 更新计数
                if response:  # 如果有返回结果
                    success_count += 1
                if response and "correct" in normalized_data:  # 如果有正确性信息
                    if normalized_data["correct"]:
                        solved_count += 1
                
                # 显示进度
                accuracy = (success_count / idx) * 100 if idx > 0 else 0
                print(f"completed {idx}/{total_tasks}, "
                      f"Solved: {bool(response)}, "
                      f"number of success = {success_count}/{idx}, "
                      f"acc = {accuracy:.2f}")
                
                # 如果不保存详情，安全地删除 details 字段
                if not save_details and 'details' in self.run_details:
                    del self.run_details['details']
                
                # 保存结果
                result = {
                    "task_id": task_id,
                    "completion": response
                }
                if hasattr(self, 'run_details') and self.run_details:
                    result.update(self.run_details)
                    
                self.results.save_result(result)
                
            except Exception as e:
                print(f"处理任务 {task_id} 时出错: {str(e)}")
                continue
