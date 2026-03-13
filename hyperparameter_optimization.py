# weighted_decay 1e-5 to 1e-2
# batch_size = [16, 32, 64, 128]
# learning_rate 1e-4 to 1e-2
# optimizers = [Adam, AdamW, SGD]
# dropout 0 to 0.75
# epochs 1 to 20

import pandas as pd
import numpy as np
import cupy as cp
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.func import vmap
import time
import math

xp = cp


data = pd.read_csv("spam.csv", encoding="latin1", usecols=[0, 1])


if (np.sum(data.isna()) != 0):
    empty_rows = data.isna().any(axis=1)
    raise ValueError("Empty Cells Detected")

rows = data.shape[0]

data = data.to_numpy()
data = np.asarray(data)

num_training_rows = round(rows * 0.9)
training_data = data[:num_training_rows, :]
evaluation_data = data[num_training_rows:, :]
x_train, y_train = training_data[:, 1], training_data[:, 0]
x_eval, y_eval = evaluation_data[:, 1], evaluation_data[:, 0]

smote = SMOTE(random_state=42)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(x_train)
X, y_train = smote.fit_resample(X, y_train)

X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
y_tensor = torch.tensor((y_train=="spam"), dtype=torch.float32)

batch_sizes = [16, 32, 64, 128]
device = torch.device("cuda")
dataset = TensorDataset(X_tensor, y_tensor)
loaders = [DataLoader(dataset, batch_size=batch, shuffle=True) for batch in batch_sizes]
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)


X = vectorizer.transform(x_eval)
X, y_eval = smote.fit_resample(X, y_eval)
X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
y_tensor = torch.tensor((y_eval=="spam"), dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
eval_loader = DataLoader(dataset, batch_size=X_tensor.shape[1])

weighted_decays = xp.linspace(1e-5, 1e-2, 20)
learning_rates = xp.linspace(1e-4, 1e-2, 20)
optimizers = np.asarray(["Adam", "AdamW", "SGD"])
drop_rates = xp.arange(15)*0.05

class SpamDetector(nn.Module):
    def __init__(self, dropouts, input_size):
        super(SpamDetector, self).__init__()
        self.M = len(dropouts)
        self.weights = nn.ParameterList([torch.empty(input_size) for _ in range(self.M)])
        self.biases = nn.ParameterList([torch.empty(1) for _ in range(self.M)])
        self.register_buffer("dropout_probs", torch.Tensor(dropouts))
        for w in self.weights:
            nn.init.uniform_(w, a=-math.sqrt(5), b=math.sqrt(5))
        bound = 1/math.sqrt(input_size)
        for b in self.biases:
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x):
        B, F = x.shape
        x_batched = x.unsqueeze(0).expand(self.M, B, F)
        if self.training:
            keep_prob = (1.0 - self.dropouts).to(x.device).view(self.M, 1, 1)
            rand = torch.rand((self.M, B, F), device=x.device)
            mask = rand < keep_prob
            mask /= keep_prob
            x_batched = x_batched * mask
        weights = torch.stack([w for w in self.weights], dim=0).view(self.M, F, 1)
        biases = torch.stack([b for b in self.biases], dim=0).view(self.M, 1, 1)
        out = torch.matmul(x_batched, weights) + biases
        out = out.squeeze()
        return out


# shape = (batch_sizes, epochs, optimizers, weighted_decays, learning_rates, drop_rates)
_, WGT, LR, DR = xp.meshgrid(xp.array([0, 1, 2]), weighted_decays, learning_rates, drop_rates)
DR_flat = xp.ravel(DR)
model = SpamDetector(DR_flat, X_tensor.shape[1])
params = model.parameters()

params = params.reshape(DR.shape)
adam_optimizer = [[[torch.optim.Adam(params=params[0, k, j, i], weight_decay=wd, lr=lr) for _, i in enumerate(drop_rates)] for lr, j in enumerate(learning_rates)] for wd, k in enumerate(weighted_decays)]
adamw_optimizer = [[[torch.optim.AdamW(params=params[1, k, j, i], weight_decay=wd, lr=lr) for _, i in enumerate(drop_rates)] for lr, j in enumerate(learning_rates)] for wd, k in enumerate(weighted_decays)]
sgd_optimizer = [[[torch.optim.SGD(params=params[2, k, j, i], weight_decay=wd, lr=lr) for _, i in enumerate(drop_rates)] for lr, j in enumerate(learning_rates)] for wd, k in enumerate(weighted_decays)]
adam_optimizer = xp.asarray(adam_optimizer)
adamw_optimizer = xp.asarray(adamw_optimizer)
sgd_optimizer = xp.asarray(sgd_optimizer)
optimizers = xp.stack([adam_optimizer, adamw_optimizer, sgd_optimizer], axis=0)
print("beginning")
for loader in loaders:
    model1 = model.copy().to(device)
    optim = optimizers.copy().to(device)
    for epoch in range(20):
        model1.train()
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            optim.zero_grad()
            loss.backward()
            optim.step()

        model1.eval()
        for X_batch, y_batch in eval_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch).squeeze()
            y_batch = y_batch.detach().cpu()
            y_batch = xp.asarray(y_batch)
            y_pred = y_pred.detach().cpu()
            y_pred = xp.asarray(y_pred)
            y_pred = (y_pred > 0.0)
            condition = y_pred == y_batch



'''
for weighted_decay in weighted_decays:
    for batch_size in [0, 1, 2, 3]:
        for learning_rate in learning_rates:
            for optim in optimizers:
                print("new")
                if optim == "Adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weighted_decay)
                elif optim == "AdamW":
                    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weighted_decay)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weighted_decay)
                loader = loaders[batch_size]
                t1 = time.time()
                for epoch in range(20):
                    model.train()
                    for X_batch, y_batch in loader:
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)
                        y_pred = model(X_batch).squeeze()
                        loss = criterion(y_pred, y_batch)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    model.eval()
                    for X_batch, y_batch in eval_loader:
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)
                        y_pred = model(X_batch).squeeze()
                        y_batch = y_batch.detach().cpu()
                        y_batch = xp.asarray(y_batch)
                        y_pred = y_pred.detach().cpu()
                        y_pred = xp.asarray(y_pred)
                        y_pred = (y_pred > 0.0)
                        condition = y_pred == y_batch
                        print("Accuracy: ", 100 * xp.sum(condition) / xp.size(condition))
                t2 = time.time()
                print(t2 - t1)
'''