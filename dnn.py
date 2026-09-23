import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetRegressor(RegressorMixin, BaseEstimator):
    """Сеть с одним скрытым слоем, ELU и Dropout, обученная SGD на стандартизованном таргете

    Совместима с sklearn, поэтому её можно ставить в Pipeline, VotingRegressor и StackingRegressor
    """

    def __init__(self, hidden_dim=64, dropout=0.3, lr=1e-2, batch_size=32, n_epochs=400, random_state=42):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y):
        X_t = torch.tensor(np.asarray(X, dtype="float32"))
        y = np.asarray(y, dtype="float64")
        self.y_mean_, self.y_std_ = y.mean(), y.std()
        y_t = torch.tensor((y - self.y_mean_) / self.y_std_, dtype=torch.float32).unsqueeze(1)

        torch.manual_seed(self.random_state)
        self.model_ = nn.Sequential(
            nn.Linear(X_t.shape[1], self.hidden_dim), nn.ELU(), nn.Dropout(self.dropout), nn.Linear(self.hidden_dim, 1)
        )
        optimizer = torch.optim.SGD(self.model_.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        loss_fn = nn.MSELoss()
        for _ in range(self.n_epochs):
            self.model_.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss_fn(self.model_(X_batch), y_batch).backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            prediction = self.model_(torch.tensor(np.asarray(X, dtype="float32"))).numpy().ravel()
        return prediction * self.y_std_ + self.y_mean_
