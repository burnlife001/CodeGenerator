import pandas as pd
import json
from utils.jsonl import read_jsonl, write_jsonl


def gen_summary(results_path: str, summary_path: str):
    """生成结果汇总"""
    try:
        # 读取结果文件
        results = pd.read_json(results_path, lines=True)
        
        # 确保必要字段存在
        if 'is_solved' not in results.columns:
            results['is_solved'] = results['success'] if 'success' in results.columns else False
            
        # 计算统计信息
        total = len(results)
        solved = len(results[results['is_solved'] == True])
        success = len(results[results['success'] == True]) if 'success' in results.columns else solved
        
        # 生成汇总报告
        summary = f"""
结果统计
-----------------
总任务数: {total}
成功数: {success}
解决数: {solved}
成功率: {(success/total)*100:.2f}%
解决率: {(solved/total)*100:.2f}%
"""
        
        # 保存汇总报告
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
            
        print("\n" + summary)
        
    except Exception as e:
        print(f"生成汇总报告时出错: {str(e)}")
        # 创建一个基本的错误报告
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"生成汇总报告失败: {str(e)}")

