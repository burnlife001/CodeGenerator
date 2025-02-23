from utils.jsonl import read_jsonl, write_jsonl
from evaluations.func_evaluate import evaluate_io_et
import os


def generate_et_dataset_human(results_path: str, et_results_path: str):
    """生成人工评估数据集"""
    results = read_jsonl(results_path)
    et_results = []
    
    for result in results:
        # 获取生成的代码
        code = None
        if "source_codes" in result and result["source_codes"]:
            code = result["source_codes"][0]
        elif "completion" in result:
            code = result["completion"]
        else:
            print(f"警告: 结果中缺少代码内容: {result.get('task_id', '未知任务')}")
            continue
            
        et_result = {
            "task_id": result.get("task_id", ""),
            "source_codes": [code] if code else [],
            "prompt": result.get("prompt", ""),
            "is_solved": result.get("is_solved", False),
            "success": result.get("success", False)
        }
        et_results.append(et_result)
    
    # 保存评估数据集
    write_jsonl(et_results_path, et_results)
    print(f"已生成评估数据集: {et_results_path}")


def generate_et_dataset_mbpp(
        NORMAL_RESULTS_PATH,
        ET_RESULTS_PATH,
        ET_DATA_PATH="data/MBPPEval/MBPP_ET.jsonl"
):
    dataset = read_jsonl(ET_DATA_PATH)
    data_dict = {}
    for item in dataset:
        data_dict[item["task_id"]] = {"et_item": item}

    results = read_jsonl(NORMAL_RESULTS_PATH)
    for result in results:
        task_id = int(result["name"].split("_")[1])
        data_dict[task_id]["result"] = result

    correct_count = 0
    et_results = []
    for key, value in data_dict.items():
        item = value["et_item"]
        result = value.get("result", None)
        if result is None:
            continue

        generated_code = result["source_codes"][0] if "source_codes" in result else result["solution"]

        passed = evaluate_io_et(
            item['test_list'],
            generated_code
        )

        if passed:
            result["is_solved"] = True
            correct_count += 1
        else:
            result["is_solved"] = False

        et_results.append(result)
        print(
            f"Accuracy: {correct_count}/{len(et_results)} = {correct_count/len(et_results):.2f}")
    # write_jsonl(ET_RESULTS_PATH, et_results)

    et_results = sorted(
        et_results,
        key=lambda x: int(x["name"].split("_")[1])
    )

    write_jsonl(ET_RESULTS_PATH, et_results)
    print(
        f"Accuracy: {correct_count}/{len(et_results)} = {correct_count/len(et_results):.2f}")


