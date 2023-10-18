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


def load_data(file_path: str, limit: int):# -> tuple[NDArray, NDArray, NDArray, NDArray]:
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


def make_PU(tr, ca, ts) -> pd.DataFrame:
    # LP1
    tr_titles, tr_abs, tr_labels, tr_pulabels = load_data(tr, -1)
    tr_titles = tr_titles[tr_pulabels == 1]
    tr_abs = tr_abs[tr_pulabels == 1]
    tr_labels = tr_labels[tr_pulabels == 1]
    tr_pulabels = tr_pulabels[tr_pulabels == 1]
    tr_df = pd.DataFrame(
        {
            "title": tr_titles,
            "abstract": tr_abs,
            "label": tr_labels,
            "pulabel": tr_pulabels,
        }
    )
    tr_df["tr"] = 1
    tr_df["ca"] = 0
    tr_df["ts"] = 0

    # LP2
    ca_titles, ca_abs, ca_labels, ca_pulabels = load_data(ca, -1)
    ca_titles = ca_titles[ca_pulabels == 1]
    ca_abs = ca_abs[ca_pulabels == 1]
    ca_labels = ca_labels[ca_pulabels == 1]
    ca_pulabels = ca_pulabels[ca_pulabels == 1]
    ca_pulabels = 0 * ca_pulabels
    ca_df = pd.DataFrame(
        {
            "title": ca_titles,
            "abstract": ca_abs,
            "label": ca_labels,
            "pulabel": ca_pulabels,
        }
    )
    ca_df["tr"] = 0
    ca_df["ca"] = 1
    ca_df["ts"] = 0

    # LP3 + U3
    ts_titles, ts_abs, ts_labels, ts_pulabels = load_data(ts, -1)
    ts_pulabels = 0 * ts_pulabels
    ts_df = pd.DataFrame(
        {
            "title": ts_titles,
            "abstract": ts_abs,
            "label": ts_labels,
            "pulabel": ts_pulabels,
        }
    )
    ts_df["tr"] = 0
    ts_df["ca"] = 0
    ts_df["ts"] = 1

    all_df = pd.concat([tr_df, ca_df, ts_df]).reset_index(drop=True)
    # Print the size of the dataframe and the counts of true labels and PU labels
    print(f"Size of the DataFrame: {all_df.shape}")
    # print(f"Counts of PU labels: {all_df['pulabel'].value_counts()}")
    print(f"Size of the DataFrame: {tr_df.shape}")
    print(f"Counts of true labels in train set: {tr_df['label'].value_counts()}")
    print(f"Counts of pu labels in train set: {tr_df['pulabel'].value_counts()}")
    print()
    print(f"Size of the DataFrame: {ca_df.shape}")
    print(f"Counts of true labels in valid set: {ca_df['label'].value_counts()}")
    print(f"Counts of pu labels in valid set: {ca_df['pulabel'].value_counts()}")
    print()
    print(f"Size of the DataFrame: {ts_df.shape}")
    print(f"Counts of true labels in test set: {ts_df['label'].value_counts()}")
    print(f"Counts of pu labels in test set: {ts_df['pulabel'].value_counts()}")
    print()

    return all_df


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
