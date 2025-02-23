import re

def sanitize_path(path: str) -> str:
    """清理路径中的非法字符"""
    # 替换Windows文件系统不允许的字符
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, '_', path)

def get_result_path(dataset: str, strategy: str, model: str) -> str:
    """生成结果目录路径"""
    clean_model = sanitize_path(model)
    return f"results/{dataset}/{strategy}/{clean_model}"
