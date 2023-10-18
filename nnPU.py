import torch
import torch.nn as nn
import torch.nn.functional as F

class PULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=False):
        super(PULoss,self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss#lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = 0
        self.min_count = torch.tensor(1.)
    
    def forward(self, inp, target, test=False):
        assert(inp.shape == target.shape)        
        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
        if inp.is_cuda:
            if isinstance(self.min_count, float):
                self.min_count = torch.tensor(self.min_count, device=inp.device)
            else:
                self.min_count = self.min_count.clone().detach()

            if isinstance(self.prior, float):
                self.prior = torch.tensor(self.prior, device=inp.device)
            else:
                self.prior = self.prior.clone().detach()

        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count, torch.sum(unlabeled))
        
        y_positive = self.loss_func(positive*inp) * positive
        y_positive_inv = self.loss_func(-positive*inp) * positive
        y_unlabeled = self.loss_func(-unlabeled*inp) * unlabeled

        positive_risk = self.prior * torch.sum(y_positive)/ n_positive
        negative_risk = - self.prior *torch.sum(y_positive_inv)/ n_positive + torch.sum(y_unlabeled)/n_unlabeled

        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk
        else:
            return positive_risk+negative_risk

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
        positive, unlabeled = (t == self.positive).float(), (t == self.unlabeled).float()
        n_positive, n_unlabeled = max(1., positive.sum().item()), max(1., unlabeled.sum().item())

        y_positive = self.loss(x)  # per sample positive risk
        y_unlabeled = self.loss(-x)  # per sample negative risk

        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum((unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)

        if self.nnPU:
            if negative_risk.item() < -self.beta:
                objective = (positive_risk
                             - self.beta + self.gamma * negative_risk).detach() - self.gamma * negative_risk
            else:
                objective = positive_risk + negative_risk
        else:
            objective = positive_risk + negative_risk

        return objective


def train(model, device, train_loader, optimizer, prior, epoch, log_interval):
    model.train()
    tr_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss_fct = PULoss(prior=prior)
        loss = loss_fct(output.view(-1), target.type(torch.float))
        tr_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print("Train loss: ", tr_loss)


def test(model, device, test_loader, prior):
    """Testing"""
    model.eval()
    test_loss = 0
    correct = 0
    num_pos = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss_func = PULoss(prior = prior)
            test_loss += test_loss_func(output.view(-1), target.type(torch.float)).item() # sum up batch loss
            pred = torch.where(output < 0, torch.tensor(-1, device=device), torch.tensor(1, device=device)) 
            num_pos += torch.sum(pred == 1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Percent of examples predicted positive: ', float(num_pos)/len(test_loader.dataset), '\n')
    
class PUModel(nn.Module):
    """
    Basic Multi-layer perceptron as described in "Positive-Unlabeled Learning with Non-Negative Risk Estimator"
    """
    def __init__(self, input_dim, hidden_size):
        super(PUModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc5 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x

class EnergyNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(EnergyNet, self).__init__()

        self.first = nn.Linear(input_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.last = nn.Linear(hidden_dim, output_dim)
        self.b1 = nn.BatchNorm1d(hidden_dim)
        self.b2 = nn.BatchNorm1d(hidden_dim)
        self.b3 = nn.BatchNorm1d(hidden_dim)
        self.b4 = nn.BatchNorm1d(hidden_dim)
        self.leaky_relu = nn.LeakyReLU()
        # self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.b1(self.first(x))
        x = self.leaky_relu(x)
        # x = self.dropout(x)
        x = self.b2(self.linear1(x))
        x = self.leaky_relu(x)
        # x = self.dropout(x)
        x = self.b3(self.linear2(x))
        x = self.leaky_relu(x)
        # x = self.dropout(x)
        x = self.b4(self.linear3(x))
        x = self.leaky_relu(x)
        # x = self.dropout(x)
        x = self.last(x)
        # return torch.sigmoid(x)
        return x
