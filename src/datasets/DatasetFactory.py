from datasets.Dataset import Dataset
from datasets.MBPPDataset import MBPPDataset
from datasets.APPSDataset import APPSDataset
from datasets.XCodeDataset import XCodeDataset
from datasets.HumanEvalDataset import HumanDataset
from datasets.CodeContestDataset import CodeContestDataset


class DatasetFactory:
    @staticmethod
    def get_dataset_class(dataset_name):
        dataset_name = dataset_name.lower()
        if dataset_name == "apps":
            return APPSDataset
        elif dataset_name == "mbpp":
            return MBPPDataset
        elif dataset_name == "xcode":
            return XCodeDataset
        elif dataset_name == "xcodeeval":
            return XCodeDataset
        elif dataset_name == "humaneval":
            return HumanDataset
        elif dataset_name == "human":
            return HumanDataset
        elif dataset_name == "cc":
            return CodeContestDataset
        else:
            raise Exception(f"Unknown dataset name {dataset_name}")
