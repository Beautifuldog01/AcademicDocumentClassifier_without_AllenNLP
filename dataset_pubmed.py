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
            if limit > 0 and count >= limit:
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

    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range


class BiDataset_val(torch.utils.data.Dataset):
    def __init__(self, data, label, real_label):
        self.data = data
        self.label = label
        self.real_label = real_label

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.real_label[index]

    def __len__(self):
        return len(self.data)

    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range


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

        self.total_instances = len(self.all_positive_indices) + len(
            self.all_negative_indices
        )

    def __iter__(self):
        total_batches = len(self.dataset) // self.batch_size
        smaller_class_len = min(
            len(self.all_positive_indices), len(self.all_negative_indices)
        )

        # Calculate the number of positive samples per batch based on the ratio in the dataset
        num_positive_per_batch = max(
            1, round((smaller_class_len / self.total_instances) * self.batch_size)
        )
        num_negative_per_batch = self.batch_size - num_positive_per_batch

        # Backup for reusing samples from the smaller class
        positive_backup = list(self.all_positive_indices)
        negative_backup = list(self.all_negative_indices)

        # Counter for the number of cycles the smaller class has been through
        cycle_counter = self.num_cycles

        for i in range(total_batches):
            # Replenish the smaller class samples if necessary
            if num_positive_per_batch > len(self.all_positive_indices):
                random.shuffle(positive_backup)
                self.all_positive_indices += positive_backup
                cycle_counter -= 1

            if num_negative_per_batch > len(self.all_negative_indices):
                random.shuffle(negative_backup)
                self.all_negative_indices += negative_backup
                cycle_counter -= 1

            if cycle_counter == 0:
                break

            # Create a balanced batch
            num_positive_per_batch = min(
                num_positive_per_batch, len(self.all_positive_indices)
            )
            if num_positive_per_batch > 0:
                positive_indices = random.sample(
                    self.all_positive_indices, num_positive_per_batch
                )
            self.all_positive_indices = [
                x for x in self.all_positive_indices if x not in positive_indices
            ]

            num_negative_per_batch = min(
                num_negative_per_batch, len(self.all_negative_indices)
            )
            if num_negative_per_batch > 0:
                negative_indices = random.sample(
                    self.all_negative_indices, num_negative_per_batch
                )
            self.all_negative_indices = [
                x for x in self.all_negative_indices if x not in negative_indices
            ]

            batch_indices = positive_indices + negative_indices
            random.shuffle(batch_indices)

            for index in batch_indices:
                yield index

    def __len__(self):
        return len(self.dataset) // self.batch_size
