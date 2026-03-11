import pandas as pd
import numpy as np
import cupy as cp
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

xp = np

class SpamDetector(nn.Module):
    def __init__(self, input_size):
        super(SpamDetector, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


data = pd.read_csv("spam.csv", encoding="latin1", usecols=[0, 1])


if (xp.sum(data.isna()) != 0):
    empty_rows = data.isna().any(axis=1)
    raise ValueError("Empty Cells Detected")

rows = data.shape[0]
print("Rows: ", rows)
condition = data["v1"] == "spam"
print("Spam Data: ", xp.sum(condition))
print("Nonspam Data: ", xp.sum(~condition), "\n")

data = data.to_numpy()
data = xp.asarray(data)

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

device = torch.device("cuda")
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
model = SpamDetector(input_size=X_tensor.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


model = model.to(device)
criterion = criterion.to(device)
model.train()
for epoch in range(3):
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch: ", epoch)
    print("Loss: ", loss.item(), "\n")

X = vectorizer.transform(x_eval)
X, y_eval = smote.fit_resample(X, y_eval)
X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
y_tensor = torch.tensor((y_eval=="spam"), dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=X_tensor.shape[1])
model.eval()
for X_batch, y_batch in loader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    y_pred = model(X_batch).squeeze()
    y_batch = y_batch.detach().cpu()
    y_batch = xp.asarray(y_batch)
    y_pred = y_pred.detach().cpu()
    y_pred = xp.asarray(y_pred)
    y_pred = (y_pred > 0.5)
    condition = y_pred == y_batch
    print("Accuracy: ", 100*xp.sum(condition)/xp.size(condition))