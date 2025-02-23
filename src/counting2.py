import nest_asyncio
nest_asyncio.apply()
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination,TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from rich.console import Console
from rich.panel import Panel
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from web_request.client import get_client

console = Console()

async def main() -> None:
    model_client = get_client("tencent_ds_v3")#ollama_3b, tencent_ds_v3

    agent1 = AssistantAgent("Assistant1", model_client=model_client)
    agent2 = AssistantAgent("Assistant2", model_client=model_client)

    termination = MaxMessageTermination(11) | TextMentionTermination("TERMINATE")

    team = RoundRobinGroupChat([agent1, agent2], termination_condition=termination)

    stream = team.run_stream(task="Count from 1 to 10, respond one character at a time.")
    
    async for message in stream:
        if hasattr(message, 'content'):
            # Print message in panel with source as title
            console.print(Panel(
                message.content,
                title=f"[bold blue]{message.source}[/bold blue]",
                border_style="blue"
            ))
            
            # Print usage statistics if available
            if message.models_usage:
                console.print(f"[dim]Usage - Prompt tokens: {message.models_usage.prompt_tokens}, "
                         f"Completion tokens: {message.models_usage.completion_tokens}[/dim]")
            console.print("―" * 80)  # Separator line
        else:  # TaskResult object
            console.print("\n[bold yellow]Task Result Summary[/bold yellow]")
            console.print(f"Stop Reason: {message.stop_reason}")
            console.print("―" * 80)

asyncio.run(main())
