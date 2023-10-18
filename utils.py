import random
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from nltk.tokenize import word_tokenize
from collections import Counter
import itertools


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def r4(x):
    return round(x, 4)


def get_reduction(preds, labels):
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    recall = recall_score(labels, preds)
    N = len(labels)
    work_reduction = (tn + fn) / (N * (1 - recall + 1e-8))

    return work_reduction


def calculate_accuracy(y_true, y_pred):
    # (TP + TN) / (TP + TN + FP + FN)
    TP = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    TN = np.sum(np.logical_and(y_true == -1, y_pred == -1))
    FP = np.sum(np.logical_and(y_true == -1, y_pred == 1))
    FN = np.sum(np.logical_and(y_true == 1, y_pred == -1))
    return (TP + TN) / (TP + TN + FP + FN)


def only_positive(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return f1, recall


def get_threshold(logits, labels):
    threshold = 0.0
    max_f1 = 0.0
    for t in np.arange(-0.5, 0.5, 0.01):
        pred = torch.gt(logits, t).float()
        # # pos only
        # f1 = f1_score(labels, pred.squeeze(), pos_label=1)

        # pos+neg
        f1 = f1_score(labels, pred.squeeze())

        if f1 > max_f1:
            max_f1 = f1
            threshold = t
    return threshold


def get_threshold_cqy(logits, labels, n_min, p_max, step=0.001, patience=20):
    thresholds = np.arange(n_min, p_max, step)

    # Vectorized prediction
    preds = torch.gt(
        logits.unsqueeze(-1), torch.from_numpy(thresholds).float().to(logits.device)
    ).float()

    # Vectorized F1 score computation
    # # pos-only
    # f1_scores = [f1_score(labels, pred.squeeze()) for pred in preds.T]

    f1_scores = [f1_score(labels, pred.squeeze()) for pred in preds.T]

    # Early stopping
    no_improve = 0
    max_f1 = 0.0
    best_threshold = n_min
    for i, f1 in enumerate(f1_scores):
        if f1 > max_f1:
            max_f1 = f1
            best_threshold = thresholds[i]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_threshold


def get_threshold_99(probs, labels):
    positive_probs = probs[labels == 1]
    threshold99 = np.quantile(positive_probs, 0.05)
    return threshold99


def get_metric(labels, prob, threshold99=None):
    # Move tensors to CPU if they are on GPU
    labels = (
        labels.cpu() if isinstance(labels, torch.Tensor) and labels.is_cuda else labels
    )
    prob = prob.cpu() if isinstance(prob, torch.Tensor) and prob.is_cuda else prob

    # Convert to NumPy if they are tensors
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    prob = prob.numpy() if isinstance(prob, torch.Tensor) else prob
    auc = roc_auc_score(labels, prob)
    r_10, r_20, r_30, r_40, r_50, r_95 = get_rec(labels, prob)

    p_mean = np.mean(prob[labels == 1])
    n_mean = np.mean(prob[labels == 0])

    prob = torch.tensor(prob)
    threshold = get_threshold_cqy(prob, labels, n_mean, p_mean, step=0.001)
    threshold99 = get_threshold_99(prob, labels)
    preds = torch.gt(prob, threshold).float()
    preds99 = torch.gt(prob, threshold99).float()
    f1, rec = only_positive(labels, preds)
    acc = accuracy_score(labels, preds)

    f1_99, rec_99 = only_positive(labels, preds99)
    acc_99 = accuracy_score(labels, preds99)

    reduce_work = get_reduction(preds99, labels)

    return (
        threshold,
        threshold99,
        auc,
        f1,
        acc,
        rec,
        f1_99,
        acc_99,
        rec_99,
        r_10,
        r_20,
        r_30,
        r_40,
        r_50,
        r_95,
        reduce_work,
        p_mean,
        n_mean,
    )


def print_info(info_tuple):
    (
        threshold,
        threshold99,
        auc,
        f1,
        acc,
        rec,
        f1_99,
        acc_99,
        rec_99,
        r_10,
        r_20,
        r_30,
        r_40,
        r_50,
        r_95,
        reduce_work,
        p_mean,
        n_mean,
    ) = info_tuple
    print("Gold auc:", r4(auc))
    print("f1, acc, rec: ", r4(f1), r4(acc), r4(rec))
    print("f1_99, acc_99, rec_99: ", r4(f1_99), r4(acc_99), r4(rec_99))
    print("top@: ", r4(r_10), r4(r_20), r4(r_30), r4(r_40), r4(r_50), r4(r_95))
    print("reduce workload: ", r4(reduce_work))
    print("threshold: ", r4(threshold))
    print("threshold99: ", r4(threshold99))
    print("Gold positive mean:", r4(p_mean))
    print("Gold negative mean:", r4(n_mean))
    print()


def get_rec(labels, logits):
    pos_logits = logits[labels == 1]
    sorted_logits = sorted(logits, reverse=True)
    s_p = sorted(pos_logits)

    TOP10 = sorted_logits[int(len(sorted_logits) * 0.1) - 1]
    TOP20 = sorted_logits[int(len(sorted_logits) * 0.2) - 1]
    TOP30 = sorted_logits[int(len(sorted_logits) * 0.3) - 1]
    TOP40 = sorted_logits[int(len(sorted_logits) * 0.4) - 1]
    TOP50 = sorted_logits[int(len(sorted_logits) * 0.5) - 1]
    TOP95 = sorted_logits[int(len(sorted_logits) * 0.95) - 1]
    r_10 = sum(1 for i in s_p if i >= TOP10) / len(s_p)
    r_20 = sum(1 for i in s_p if i >= TOP20) / len(s_p)
    r_30 = sum(1 for i in s_p if i >= TOP30) / len(s_p)
    r_40 = sum(1 for i in s_p if i >= TOP40) / len(s_p)
    r_50 = sum(1 for i in s_p if i >= TOP50) / len(s_p)
    r_95 = sum(1 for i in s_p if i >= TOP95) / len(s_p)
    return r_10, r_20, r_30, r_40, r_50, r_95


def calculate_accuracy(y_true, y_pred):
    # (TP + TN) / (TP + TN + FP + FN)
    TP = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    TN = np.sum(np.logical_and(y_true == -1, y_pred == -1))
    FP = np.sum(np.logical_and(y_true == -1, y_pred == 1))
    FN = np.sum(np.logical_and(y_true == 1, y_pred == -1))
    return (TP + TN) / (TP + TN + FP + FN)


def getFeatures(data, word_to_index, max_length):
    all_features = []

    for index in range(len(data)):
        title_tokens = data.title[index].split()
        abstract_tokens = data.abstract[index].split()

        # Convert words to indices
        title_indices = [word_to_index.get(word.lower(), 0) for word in title_tokens]
        abstract_indices = [
            word_to_index.get(word.lower(), 0) for word in abstract_tokens
        ]

        # Pad or truncate
        title_indices += [0] * (max_length - len(title_indices))
        title_indices = title_indices[:max_length]
        abstract_indices += [0] * (max_length - len(abstract_indices))
        abstract_indices = abstract_indices[:max_length]

        all_features.append((title_indices, abstract_indices))

    return all_features


def log_metrics(writer, phase, metrics, epoch):
    """
    Log metrics using TensorBoard.
    """
    (
        threshold,
        threshold99,
        auc,
        f1,
        acc,
        rec,
        f1_99,
        acc_99,
        rec_99,
        r_10,
        r_20,
        r_30,
        r_40,
        r_50,
        r_95,
        reduce_work,
        p_mean,
        n_mean,
    ) = metrics

    writer.add_scalar(f"{phase}/AUC", auc, epoch)
    writer.add_scalar(f"{phase}/F1", f1, epoch)
    writer.add_scalar(f"{phase}/Accuracy", acc, epoch)
    writer.add_scalar(f"{phase}/Recall", rec, epoch)
    writer.add_scalar(f"{phase}/F1_99", f1_99, epoch)
    writer.add_scalar(f"{phase}/Accuracy_99", acc_99, epoch)
    writer.add_scalar(f"{phase}/Recall_99", rec_99, epoch)
    writer.add_scalar(f"{phase}/R_10", r_10, epoch)
    writer.add_scalar(f"{phase}/R_20", r_20, epoch)
    writer.add_scalar(f"{phase}/R_30", r_30, epoch)
    writer.add_scalar(f"{phase}/R_40", r_40, epoch)
    writer.add_scalar(f"{phase}/R_50", r_50, epoch)
    writer.add_scalar(f"{phase}/R_95", r_95, epoch)
    writer.add_scalar(f"{phase}/Reduce_Work", reduce_work, epoch)
    writer.add_scalar(f"{phase}/Positive_Mean", p_mean, epoch)
    writer.add_scalar(f"{phase}/Negative_Mean", n_mean, epoch)


def build_vocab(texts, min_freq=2):
    """
    Build vocabulary from a list of texts
    """
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    counter = Counter(itertools.chain.from_iterable(tokenized_texts))

    vocab = {
        word: i + 2
        for i, (word, freq) in enumerate(counter.items())
        if freq >= min_freq
    }

    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    return vocab
