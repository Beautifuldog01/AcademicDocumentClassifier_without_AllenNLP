import torch
import torch.nn as nn
import torch.nn.functional as F

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
