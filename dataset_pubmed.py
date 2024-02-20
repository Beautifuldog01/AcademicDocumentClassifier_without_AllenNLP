import json
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Sampler


def parse_pubmed_json(pubmed_json):
    title = pubmed_json["title"]
    abstract = pubmed_json["abstract"]
    true_label = 0 if pubmed_json["label_true"] == "negative/unlabeled" else 1

    label_L100 = pubmed_json.get("label_L100")
    label_L40 = pubmed_json.get("label_L40")
    PU_label = (
        0
        if (label_L100 == "negative/unlabeled" or label_L40 == "negative/unlabeled")
        else 1
    )

    return title, abstract, true_label, PU_label


def load_data(
    file_path: str, limit: int
):  # -> tuple[NDArray, NDArray, NDArray, NDArray]:
    count = 0
    data = {"titles": [], "abstracts": [], "true_labels": [], "PU_labels": []}

    with open(file_path, "r") as data_file:
        for line in data_file:
            if 0 < limit <= count:
                break
            pubmed_json = json.loads(line)
            title, abstract, true_label, PU_label = parse_pubmed_json(pubmed_json)

            data["titles"].append(title)
            data["abstracts"].append(abstract)
            data["true_labels"].append(true_label)
            data["PU_labels"].append(PU_label)

            count += 1

    return (
        np.array(data["titles"]),
        np.array(data["abstracts"]),
        np.array(data["true_labels"]),
        np.array(data["PU_labels"]),
    )


def load_and_transform_data(data, tr=0, ca=0, ts=0):
    titles, abs, labels, pulabels = load_data(data, -1)
    df = pd.DataFrame(
        {
            "title": titles,
            "abstract": abs,
            "label": labels,
            "pulabel": pulabels,
            "tr": tr,
            "ca": ca,
            "ts": ts,
        }
    )
    return df


def make_PU_meta(tr, ca, ts):
    tr_df = load_and_transform_data(tr, tr=1)
    ca_df = load_and_transform_data(ca, ca=1)
    ts_df = load_and_transform_data(ts, ts=1)

    all_df = pd.concat([tr_df, ca_df, ts_df]).reset_index(drop=True)

    for label, df in [("Training", tr_df), ("Valid", ca_df), ("Test", ts_df)]:
        print(f"Size of the {label} DataFrame: {df.shape}")
        print(
            f"Counts of true labels in {label.lower()} set: {df['label'].value_counts()}"
        )
        print(
            f"Counts of pu labels in {label.lower()} set: {df['pulabel'].value_counts()}"
        )
        print()

    return all_df


class BiDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def normalization(self):
        _range = np.max(self.data) - np.min(self.data)
        return (self.data - np.min(self.data)) / _range


class BiDataset_val(torch.utils.data.Dataset):
    def __init__(self, data, label, real_label):
        self.data = data
        self.label = label
        self.real_label = real_label

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.real_label[index]

    def __len__(self):
        return len(self.data)

    def normalization(self):
        _range = np.max(self.data) - np.min(self.data)
        return (self.data - np.min(self.data)) / _range


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, positive_ratio=0.5):
        self.dataset = dataset
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        self.all_positive_indices = [
            i for i, label in enumerate(self.dataset.label) if label == 1
        ]
        self.all_negative_indices = [
            i for i, label in enumerate(self.dataset.label) if label == 0
        ]

    def __iter__(self):
        total_batches = len(self.dataset) // self.batch_size
        num_positive_per_batch = int(self.batch_size * self.positive_ratio)
        num_negative_per_batch = self.batch_size - num_positive_per_batch

        for i in range(total_batches):
            positive_indices = random.choices(
                self.all_positive_indices, k=num_positive_per_batch
            )
            negative_indices = random.choices(
                self.all_negative_indices, k=num_negative_per_batch
            )
            batch_indices = positive_indices + negative_indices
            random.shuffle(batch_indices)

            for index in batch_indices:
                yield index

    def __len__(self):
        return len(self.dataset) // self.batch_size

class ProportionalSampler(Sampler):
    def __init__(self, dataset, batch_size, num_cycles):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_cycles = num_cycles

        self.all_positive_indices = [
            i for i, label in enumerate(self.dataset.label) if label == 1
        ]
        self.all_negative_indices = [
            i for i, label in enumerate(self.dataset.label) if label == 0
        ]

        self.total_instances = len(self.all_positive_indices) + len(self.all_negative_indices)
        self.smaller_class, self.larger_class = (
            (set(self.all_positive_indices), set(self.all_negative_indices))
            if len(self.all_positive_indices) < len(self.all_negative_indices)
            else (set(self.all_negative_indices), set(self.all_positive_indices))
        )

    def __iter__(self):
        cycle_counter = self.num_cycles
        total_batches = len(self.dataset) // self.batch_size
        used_smaller_class_indices = set()

        for _ in range(total_batches):
            if cycle_counter > 0:
                num_smaller_per_batch = max(
                    1, round((len(self.smaller_class) / self.total_instances) * self.batch_size)
                )
                
                if len(self.smaller_class) - len(used_smaller_class_indices) < num_smaller_per_batch:
                    used_smaller_class_indices = set()
                
                available_smaller_class_indices = list(self.smaller_class - used_smaller_class_indices)
                smaller_class_indices = random.sample(available_smaller_class_indices, num_smaller_per_batch)
                used_smaller_class_indices.update(smaller_class_indices)
                
                num_larger_per_batch = self.batch_size - num_smaller_per_batch
                larger_class_indices = random.sample(list(self.larger_class), num_larger_per_batch)
                
                batch_indices = smaller_class_indices + larger_class_indices
                random.shuffle(batch_indices)
                
                if len(used_smaller_class_indices) == len(self.smaller_class):
                    cycle_counter -= 1
                    used_smaller_class_indices = set()

            else:
                batch_indices = random.sample(
                    list(self.smaller_class) + list(self.larger_class), self.batch_size
                )

            for index in batch_indices:
                yield index

    def __len__(self):
        return len(self.dataset) // self.batch_size