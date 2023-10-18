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
    def __init__(self, dataset):
        self.dataset = dataset
        self.all_positive_indices = [
            i for i, label in enumerate(self.dataset.label) if label == 1
        ]
        self.all_negative_indices = [
            i for i, label in enumerate(self.dataset.label) if label == 0
        ]

    def __iter__(self):
        random.shuffle(self.all_positive_indices)
        random.shuffle(self.all_negative_indices)

        len_smaller = min(
            len(self.all_positive_indices), len(self.all_negative_indices)
        )

        for i in range(len_smaller):
            yield self.all_positive_indices[i]
            yield self.all_negative_indices[i]

        remaining_indices = (
            self.all_positive_indices[len_smaller:]
            + self.all_negative_indices[len_smaller:]
        )
        random.shuffle(remaining_indices)

        for i in remaining_indices:
            yield i

    def __len__(self):
        return len(self.dataset)
