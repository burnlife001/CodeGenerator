import dotenv
dotenv.load_dotenv()  # 加载环境变量

import argparse
import sys
from datetime import datetime
from constants.paths import *

from models.Gemini import Gemini
from models.OpenAI import OpenAIModel
from models.TencentCloud import TencentCloudModel  # 添加腾讯云模型导入

from results.Results import Results

from promptings.PromptingFactory import PromptingFactory
from datasets.DatasetFactory import DatasetFactory
from models.ModelFactory import ModelFactory

from constants.verboseType import *

from utils.summary import gen_summary
from utils.runEP import run_eval_plus
from utils.evaluateET import generate_et_dataset_human
from utils.evaluateET import generate_et_dataset_mbpp
from utils.generateEP import generate_ep_dataset_human
from utils.generateEP import generate_ep_dataset_mbpp
from utils.path_utils import get_result_path

# 设置命令行参数解析器
parser = argparse.ArgumentParser()

# 设置数据集参数
parser.add_argument(
    "--dataset",
    type=str,
    default="HumanEval",
    choices=[
        "HumanEval",  # 164个手写编程问题数据集
        "MBPP",       # MBPP实际编程场景数据集
        "APPS",       # APPS复杂算法数据集
        "xCodeEval",  # 单测执行数据集
        "CC",         # 巧合测试数据集
    ]
)

# 设置代码生成策略
parser.add_argument(
    "--strategy",
    type=str,
    default="Direct",
    choices=[
        "Direct",        # 直接生成
        "CoT",          # 思维链
        "SelfPlanning", # 自我规划
        "Analogical",   # 类比推理
        "MapCoder",     # 映射编码
        "CodeSIM",      # 基础版CodeSIM模型
        "CodeSIMWD",
        "CodeSIMWPV",
        "CodeSIMWPVD",
        "CodeSIMA",
        "CodeSIMC",
    ]
)

parser.add_argument(
    "--model",
    type=str,
    default="ChatGPT",
)
parser.add_argument(
    "--model_provider",
    type=str,
    default="OpenAI",
    choices=[
        "OpenAI",
        "TencentCloud",  # 添加腾讯云选项
        "Gemini",
        "ollama"        # 添加 ollama 选项
    ]
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0.95
)
parser.add_argument(
    "--pass_at_k",
    type=int,
    default=1
)
parser.add_argument(
    "--language",
    type=str,
    default="Python3",
    choices=[
        "C",
        "C#",
        "C++",
        "Go",
        "PHP",
        "Python3",
        "Ruby",
        "Rust",
    ]
)

parser.add_argument(
    "--cont",
    type=str,
    default="yes",
    choices=[
        "yes",
        "no"
    ]
)

parser.add_argument(
    "--result_log",
    type=str,
    default="partial",
    choices=[
        "full",
        "partial"
    ]
)

parser.add_argument(
    "--verbose",
    type=str,
    default="2",
    choices=[
        "2",
        "1",
        "0",
    ]
)

parser.add_argument(
    "--store_log_in_file",
    type=str,
    default="yes",
    choices=[
        "yes",
        "no",
    ]
)

args = parser.parse_args()

# 添加运行参数提示
print("\n====== 运行配置信息 ======")
print(f"数据集: {args.dataset}")
print(f"生成策略: {args.strategy}")
print(f"模型提供商: {args.model_provider}")
print(f"模型名称: {args.model}")
print(f"采样温度: {args.temperature}")
print(f"Top P: {args.top_p}")
print(f"Pass@k: {args.pass_at_k}")
print(f"编程语言: {args.language}")
print("========================\n")

DATASET = args.dataset
STRATEGY = args.strategy
MODEL_NAME = args.model
MODEL_PROVIDER_NAME = args.model_provider
TEMPERATURE = args.temperature
TOP_P = args.top_p
PASS_AT_K = args.pass_at_k
LANGUAGE = args.language
CONTINUE = args.cont
RESULT_LOG_MODE = args.result_log
VERBOSE = int(args.verbose)
STORE_LOG_IN_FILE = args.store_log_in_file

# 特殊处理模型名称
if MODEL_PROVIDER_NAME == "TencentCloud":
    if MODEL_NAME == "ChatGPT":  # 如果默认是 ChatGPT，改为腾讯云默认模型
        MODEL_NAME = "deepseek-v3"
        print(f"注意: 使用腾讯云时已将默认模型更改为 {MODEL_NAME}")
    MODEL_NAME_FOR_RUN = f"TC-{MODEL_NAME}"  # 在运行名称中添加 TC 前缀
elif MODEL_PROVIDER_NAME == "ollama":  # 添加 ollama 的特殊处理
    if MODEL_NAME == "ChatGPT":
        MODEL_NAME = "llama3.2:3b"
        print(f"注意: 使用 Ollama 时已将默认模型更改为 {MODEL_NAME}")
    MODEL_NAME_FOR_RUN = f"OL-{MODEL_NAME}"
else:
    MODEL_NAME_FOR_RUN = MODEL_NAME

RUN_NAME = get_result_path(DATASET, STRATEGY, MODEL_NAME_FOR_RUN)

run_no = 1
while os.path.exists(f"{RUN_NAME}/Run-{run_no}"):
    run_no += 1

if CONTINUE == "yes" and run_no > 1:
    run_no -= 1

RUN_NAME = f"{RUN_NAME}/Run-{run_no}"

if not os.path.exists(RUN_NAME):
    os.makedirs(RUN_NAME)
    print(f"创建运行目录: {RUN_NAME}")

RESULTS_PATH = f"{RUN_NAME}/Results.jsonl"
SUMMARY_PATH = f"{RUN_NAME}/Summary.txt"
LOGS_PATH = f"{RUN_NAME}/Log.txt"

print(f"结果将保存到: {RESULTS_PATH}")
if STORE_LOG_IN_FILE.lower() == 'yes':
    print(f"日志将保存到: {LOGS_PATH}")
    sys.stdout = open(
        LOGS_PATH,
        mode="a",
        encoding="utf-8"
    )

# 创建一个同时写入文件和控制台的类
class TeeStream:
    def __init__(self, stdout, file_stream):
        self.stdout = stdout
        self.file_stream = file_stream
        self.encoding = stdout.encoding

    def write(self, data):
        self.stdout.write(data)
        self.file_stream.write(data)
        # 确保实时刷新输出
        self.stdout.flush()
        self.file_stream.flush()

    def flush(self):
        self.stdout.flush()
        self.file_stream.flush()

# 修改日志输出处理
if STORE_LOG_IN_FILE.lower() == 'yes':
    log_file = open(LOGS_PATH, mode="a", encoding="utf-8")
    # 使用TeeStream替代直接重定向
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    print(f"日志将同时输出到控制台和文件: {LOGS_PATH}")

if CONTINUE == "no" and VERBOSE >= VERBOSE_MINIMAL:
    print(f"""
##################################################
Experiment start {RUN_NAME}, Time: {datetime.now()}
###################################################
""")

# 实验运行主流程
print("\n开始执行代码生成...")
print(f"时间: {datetime.now()}")
print("------------------------")

strategy = PromptingFactory.get_prompting_class(STRATEGY)(
    model=ModelFactory.get_model_class(MODEL_PROVIDER_NAME)(
        model_name=MODEL_NAME, 
        temperature=TEMPERATURE,  # 采样温度 
        top_p=TOP_P              # 核采样阈值
    ),
    data=DatasetFactory.get_dataset_class(DATASET)(),
    language=LANGUAGE,
    pass_at_k=PASS_AT_K,        # k次通过评估
    results=Results(RESULTS_PATH),
    verbose=VERBOSE
)

# 执行代码生成
strategy.run(RESULT_LOG_MODE.lower() == 'full')

print("------------------------")
print(f"代码生成完成")
print(f"时间: {datetime.now()}")

if VERBOSE >= VERBOSE_MINIMAL:
    print(f"""
##################################################
Experiment end {RUN_NAME}, Time: {datetime.now()}
###################################################
""")

# 生成实验总结
gen_summary(RESULTS_PATH, SUMMARY_PATH)

# 生成额外评估数据集
ET_RESULTS_PATH = f"{RUN_NAME}/Results-ET.jsonl"
ET_SUMMARY_PATH = f"{RUN_NAME}/Summary-ET.txt"

EP_RESULTS_PATH = f"{RUN_NAME}/Results-EP.jsonl"
EP_SUMMARY_PATH = f"{RUN_NAME}/Summary-EP.txt"

if "human" in DATASET.lower():
    generate_et_dataset_human(RESULTS_PATH, ET_RESULTS_PATH)
    gen_summary(ET_RESULTS_PATH, ET_SUMMARY_PATH)

    # generate_ep_dataset_human(RESULTS_PATH, EP_RESULTS_PATH)
    # run_eval_plus(EP_RESULTS_PATH, EP_SUMMARY_PATH, "humaneval")

elif "mbpp" in DATASET.lower():
    generate_et_dataset_mbpp(RESULTS_PATH, ET_RESULTS_PATH)
    gen_summary(ET_RESULTS_PATH, ET_SUMMARY_PATH)

    # generate_ep_dataset_human(RESULTS_PATH, EP_RESULTS_PATH)
    # run_eval_plus(EP_RESULTS_PATH, EP_SUMMARY_PATH, "mbpp")

print("\n====== 运行完成 ======")
print(f"结果文件: {RESULTS_PATH}")
print(f"总结文件: {SUMMARY_PATH}")
if "human" in DATASET.lower() or "mbpp" in DATASET.lower():
    print(f"ET评估文件: {ET_RESULTS_PATH}")
print("===================\n")

if STORE_LOG_IN_FILE.lower() == 'yes':
    sys.stdout.close()

