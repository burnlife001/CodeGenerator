import os
import time
from tenacity import retry, wait_random_exponential, stop_after_attempt
from .Base import BaseModel
from rich.console import Console
from rich.panel import Panel

console = Console()
usage_log_file_path = "usage_log.csv"

class TencentCloudModel(BaseModel):
    def __init__(
        self,
        model_name: str = "deepseek-v3",  # 模型名称
        temperature: float = 0,            # 采样温度
        top_p: float = 0.95,              # 核采样阈值
        sleep_time: int = 0,              # API 调用间隔时间
        **kwargs
    ):
        """
        初始化腾讯云模型
        Args:
            model_name: 使用的模型名称
            temperature: 生成文本的随机性(0-2之间,越大越随机)
            top_p: 核采样的概率阈值(0-1之间)
            sleep_time: API调用间隔时间(秒)
        """
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
        """创建腾讯云API客户端"""
        from openai import OpenAI
        return OpenAI(
            api_key=os.getenv("QQ_API_KEY"),
            base_url="https://api.lkeap.cloud.tencent.com/v1",
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def prompt(
        self, 
        processed_input: list[dict], 
        frequency_penalty=0, 
        presence_penalty=0
    ):
        """
        发送提示并获取响应
        Args:
            processed_input: 处理过的输入消息列表
            frequency_penalty: 频率惩罚参数
            presence_penalty: 存在惩罚参数
        Returns:
            tuple: (生成的内容, 运行详情)
        """
        time.sleep(self.sleep_time)
        
        # 显示输入消息
        self.console.print("")  # 添加空行
        for msg in processed_input:
            self.console.print(Panel(
                msg["content"],
                title=f"[bold]{msg['role'].upper()}[/bold]",
                border_style="blue"
             ))
            self.console.print("")  # 确保面板之间有间隔
        
        start_time = time.perf_counter()
        
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
        
        end_time = time.perf_counter()
        taken_time = end_time - start_time

        # 显示响应内容
        self.console.print(Panel(
            response.choices[0].message.content,
            title="[bold]ASSISTANT[/bold]",
            border_style="green"
        ))
        self.console.print("")  # 添加空行
        
        # 显示统计信息
        self.console.print(
            f"[dim]Token统计: "
            f"提示tokens={response.usage.prompt_tokens}, "
            f"完成tokens={response.usage.completion_tokens}, "
            f"总计={response.usage.total_tokens} | "
            f"响应时间: {taken_time:.2f}秒[/dim]"
        )
        self.console.print("―" * 80)  # 添加分隔线

        # 记录使用情况
        with open(usage_log_file_path, mode="a") as file:
            file.write(f'{self.model_name},{response.usage.prompt_tokens},{response.usage.completion_tokens}\n')

        run_details = {
            "api_calls": 1,
            "taken_time": taken_time,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "details": [
                {
                    "model_name": self.model_name,
                    "model_prompt": processed_input,
                    "model_response": response.choices[0].message.content,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty
                }
            ],
        }

        return response.choices[0].message.content, run_details
