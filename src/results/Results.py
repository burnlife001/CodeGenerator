import os
import json

from utils.jsonl import read_jsonl, write_jsonl, append_in_jsonl

"""
In this file, we define the Results class, 
which is used to store the results of the simulation.

It will take a result path at first and after each 
simulation, it will save the results in that path.

Results are in the form of a list of dictionaries
and will be saved as a jsonl file.
"""


class Results(object):
    def __init__(
        self, 
        result_path: str, 
        discard_previous_run: bool = False
    ):
        self.result_path = result_path
        self.discard_previous_run = discard_previous_run
        self._ensure_results_dir()
        self.load_results()

    def _ensure_results_dir(self):
        """确保结果目录存在"""
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)

    def add_result(self, result: dict):
        self.results.append(result)
        self.append_results(result)
    
    def append_results(self, result):
        append_in_jsonl(self.result_path, result)

    def save_results(self):
        write_jsonl(self.result_path, self.results)

    def save_result(self, result: dict):
        """保存单条结果"""
        try:
            with open(self.result_path, 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"保存结果时出错: {str(e)}")

    def load_results(self):
        if os.path.exists(self.result_path):
            if self.discard_previous_run:
                os.remove(self.result_path)
            else:
                self.results = read_jsonl(self.result_path)
        else:
            self.results = []

    def get_results(self):
        return self.results

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        return self.results[idx]
