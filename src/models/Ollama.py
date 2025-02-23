from typing import Dict, Any, Tuple
import os
import time
import openai
from tenacity import retry, wait_exponential, stop_after_attempt
from .Base import BaseModel
from rich.console import Console
from rich.panel import Panel

console = Console()
usage_log_file_path = "usage_log.csv"

class OllamaModel(BaseModel):
    def __init__(
        self,
        model_name: str = "llama3.2:3b",  # 默认模型
        temperature: float = 0,            
        top_p: float = 0.95,              
        sleep_time: int = 0,              
        **kwargs
    ):
        if model_name is None:
            raise Exception("Model name is required")
            
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.sleep_time = sleep_time
        
        self.client = self._create_client()
        self.console = Console()
    
    def _create_client(self):
        """创建 Ollama API 客户端"""
        return openai.OpenAI(
            api_key="ollama",  # ollama不需要api key
            base_url="http://localhost:11434/v1"
        )

    @retry(
        stop=stop_after_attempt(5),  # 最多重试5次
        wait=wait_exponential(multiplier=1, min=4, max=60),  # 指数退避重试
        reraise=True
    )
    def prompt(self, processed_input: list[dict], frequency_penalty=0, presence_penalty=0) -> Tuple[str, Dict[str, Any]]:
        """发送提示并获取响应"""
        time.sleep(self.sleep_time)
        
        # 显示所有输入消息
        self.console.print("")
        for msg in processed_input:
            self.console.print(Panel(
                msg["content"],
                title=f"[bold]{msg['role'].upper()}[/bold]",
                border_style="blue"
            ))
            self.console.print("")
        
        start_time = time.perf_counter()
        
        try:
            # API调用
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=processed_input,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=None,
                stream=False
            )
            
            content = response.choices[0].message.content
            taken_time = time.perf_counter() - start_time
            
            # 显示响应内容
            self.console.print(Panel(
                content,
                title="[bold]ASSISTANT[/bold]",
                border_style="green"
            ))
            self.console.print("")
            
            # 获取token统计
            prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
            completion_tokens = getattr(response.usage, 'completion_tokens', 0)
            total_tokens = getattr(response.usage, 'total_tokens', 0)
            
            # 显示统计信息
            self.console.print(
                f"[dim]Token统计: "
                f"提示tokens={prompt_tokens}, "
                f"完成tokens={completion_tokens}, "
                f"总计={total_tokens} | "
                f"响应时间: {taken_time:.2f}秒[/dim]"
            )
            self.console.print("―" * 80)
            
            # 记录使用情况
            with open(usage_log_file_path, mode="a") as file:
                file.write(f'{self.model_name},{prompt_tokens},{completion_tokens}\n')
            
            # 构建运行详情
            run_details = {
                "api_calls": 1,
                "taken_time": taken_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "details": [{
                    "model_name": self.model_name,
                    "model_prompt": processed_input,
                    "model_response": content,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty
                }]
            }
            
            return content, run_details
            
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            raise
        except Exception as e:
            print(f"Ollama API 调用错误: {str(e)}")
            raise
