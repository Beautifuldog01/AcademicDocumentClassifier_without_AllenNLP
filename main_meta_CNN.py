import os
import datetime
import itertools
import torch
import numpy as np
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset_pubmed import *
from checkpoint import *
from func import *
from utils import *
from nnPU import *


experiments = [
    "data/pubmed-dse/L50/D000328.D008875.D015658",
    "data/pubmed-dse/L50/D000818.D001921.D051381",
    "data/pubmed-dse/L50/D006435.D007676.D008875",
    "data/pubmed-dse/L20/D000328.D008875.D015658",
    "data/pubmed-dse/L20/D000818.D001921.D051381",
    "data/pubmed-dse/L20/D006435.D007676.D008875",
]

root_dir = experiments[0]

tr_file_path = os.path.join(root_dir, "train.jsonl")
va_file_path = os.path.join(root_dir, "valid.jsonl")
ts_file_path = os.path.join(root_dir, "test.jsonl")

all_df = make_PU_meta(tr_file_path, va_file_path, ts_file_path)
train_index = all_df.query("tr == 1").index
train_labels = all_df.query("tr == 1")["pulabel"].values
val_index = all_df.query("ca == 1").index
val_labels = all_df.query("ca == 1")["label"].values
test_index = all_df.query("ts == 1").index
test_labels = all_df.query("ts == 1")["label"].values


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


class NonNegativePULoss(nn.Module):
    def __init__(self, prior, positive_class=1, loss=None, gamma=1, beta=0, nnpu=True):
        super(NonNegativePULoss, self).__init__()
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss = loss or (lambda x: torch.sigmoid(-x))
        self.nnPU = nnpu
        self.positive = positive_class
        self.unlabeled = 1 - positive_class

    def forward(self, x, t):
        t = t[:, None]
        positive, unlabeled = (t == self.positive).float(), (
            t == self.unlabeled
        ).float()
        n_positive, n_unlabeled = max(1.0, positive.sum().item()), max(
            1.0, unlabeled.sum().item()
        )

        y_positive = self.loss(x)  # per sample positive risk
        y_unlabeled = self.loss(-x)  # per sample negative risk

        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled
        )

        if self.nnPU:
            if negative_risk.item() < -self.beta:
                objective = (
                    positive_risk - self.beta + self.gamma * negative_risk
                ).detach() - self.gamma * negative_risk
            else:
                objective = positive_risk + negative_risk
        else:
            objective = positive_risk + negative_risk

        return objective


all_df["combined_text"] = all_df["title"] + " " + all_df["abstract"]
all_texts = all_df["combined_text"].tolist()
vocab = build_vocab(all_texts)
word_to_index = {word: index for index, word in enumerate(vocab)}
all_features = getFeatures(all_df, word_to_index, max_length=500)


# Define the neural network class
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.title_cnn_3 = nn.Conv1d(embedding_dim, 50, 3, padding=1)
        self.title_cnn_5 = nn.Conv1d(embedding_dim, 50, 5, padding=2)
        self.abstract_cnn_3 = nn.Conv1d(embedding_dim, 100, 3, padding=1)
        self.abstract_cnn_5 = nn.Conv1d(embedding_dim, 100, 5, padding=2)
        self.classifier = nn.Linear(300, 1)

    def forward(self, title, abstract):
        title_embed = self.embedding(title)
        abstract_embed = self.embedding(abstract)
        title_features_3 = F.relu(self.title_cnn_3(title_embed.permute(0, 2, 1)))
        title_features_5 = F.relu(self.title_cnn_5(title_embed.permute(0, 2, 1)))
        abstract_features_3 = F.relu(
            self.abstract_cnn_3(abstract_embed.permute(0, 2, 1))
        )
        abstract_features_5 = F.relu(
            self.abstract_cnn_5(abstract_embed.permute(0, 2, 1))
        )
        title_features_3 = F.max_pool1d(
            title_features_3, title_features_3.size(2)
        ).squeeze(2)
        title_features_5 = F.max_pool1d(
            title_features_5, title_features_5.size(2)
        ).squeeze(2)
        abstract_features_3 = F.max_pool1d(
            abstract_features_3, abstract_features_3.size(2)
        ).squeeze(2)
        abstract_features_5 = F.max_pool1d(
            abstract_features_5, abstract_features_5.size(2)
        ).squeeze(2)
        title_features = torch.cat([title_features_3, title_features_5], dim=1)
        abstract_features = torch.cat([abstract_features_3, abstract_features_5], dim=1)
        combined_features = torch.cat([title_features, abstract_features], dim=1)
        output = self.classifier(combined_features)
        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassifier(len(vocab), 300).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 32
num_epochs = 50
best_va_f1 = 0
best_ts_f1 = 0
writer = SummaryWriter("runs/nnPU_CNN")
train_data = BiDataset(
    torch.tensor(all_features)[train_index], torch.tensor(train_labels)
)
train_sampler = BalancedBatchSampler(train_data)
train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)

eval_dataset = BiDataset(torch.tensor(all_features)[val_index], val_labels)
eval_dataloader = torch.utils.data.DataLoader(
    eval_dataset, batch_size=batch_size, shuffle=False
)
test_dataset = BiDataset(torch.tensor(all_features)[test_index], test_labels)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

models_dir = os.mkdir("models")
model_for_nnpu = os.path.join(models_dir, "nnPUCNN")
current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
model_for_nnpu = os.path.join(model_for_nnpu, current_date)
os.makedirs(model_for_nnpu, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fct = NonNegativePULoss(prior=0.5)

# Training loop
for epoch in tqdm(range(num_epochs)):
    total_loss = 0.0
    for i, (content, labels) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        content = content.to(device)
        labels = labels.to(device)
        outputs = model(content[:, 0, :], content[:, 1, :])
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_prob = []
    with torch.no_grad():
        for i, (eval_content, _) in enumerate(eval_dataloader):
            eval_content = eval_content.to(device)
            eval_outputs = model(eval_content[:, 0, :], eval_content[:, 1, :])
            val_prob.append(eval_outputs.squeeze().cpu().numpy())
    val_prob = np.hstack(val_prob)
    npuu_eval_info_tuple = get_metric(val_labels, val_prob)
    log_metrics(writer, "Validation", npuu_eval_info_tuple, epoch)
    npuu_val_threshold99 = npuu_eval_info_tuple[1]
    if npuu_eval_info_tuple[3] > best_va_f1:
        best_va_f1 = npuu_eval_info_tuple[3]

    test_prob = []
    with torch.no_grad():
        for i, (test_content, _) in enumerate(test_dataloader):
            test_content = test_content.to(device)
            test_outputs = model(test_content[:, 0, :], test_content[:, 1, :])
            test_prob.append(test_outputs.squeeze().cpu().numpy())
    test_prob = np.hstack(test_prob)
    npuu_test_info_tuple = get_metric(test_labels, test_prob, npuu_val_threshold99)
    log_metrics(writer, "Test", npuu_test_info_tuple, epoch)
    if npuu_test_info_tuple[3] > best_ts_f1:
        best_ts_f1 = npuu_test_info_tuple[3]
        # print("<<<<<<TEST>>>>>>")
        # print_info(npuu_test_info_tuple)
        best_model_state = model.state_dict()
        val_neg_scores = val_prob[val_labels == 0]
        val_pos_scores = val_prob[val_labels == 1]

        # plt.figure(figsize=(10, 8))
        # plt.hist(val_neg_scores, bins=50, alpha=0.5, label='Negative Samples')
        # plt.hist(val_pos_scores, bins=50, alpha=0.5, label='Positive Samples')
        # plt.legend(loc='upper right')
        # plt.title('Score Distribution for Validation Set')
        # plt.show()

        test_neg_scores = test_prob[test_labels == 0]
        test_pos_scores = test_prob[test_labels == 1]

        # plt.figure(figsize=(10, 8))
        # plt.hist(test_neg_scores, bins=50, alpha=0.5, label='Negative Samples')
        # plt.hist(test_pos_scores, bins=50, alpha=0.5, label='Positive Samples')
        # plt.legend(loc='upper right')
        # plt.title('Score Distribution for Test Set')
        # plt.show()
print_info(npuu_test_info_tuple)
torch.save(
    best_model_state,
    os.path.join(model_for_nnpu, f"npuu_model_best_test_f1_{best_ts_f1:.3f}.pth"),
)
writer.close()
print("===TRAIN NNPU END===")