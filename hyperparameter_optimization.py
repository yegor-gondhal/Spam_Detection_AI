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
import copy

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
loaders = [DataLoader(dataset, batch_size=batch, shuffle=True, pin_memory=True) for batch in batch_sizes]
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)


X = vectorizer.transform(x_eval)
X, y_eval = smote.fit_resample(X, y_eval)
X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
y_tensor = torch.tensor((y_eval=="spam"), dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
eval_loader = DataLoader(dataset, batch_size=X_tensor.shape[1], pin_memory=True)

weighted_decays = np.linspace(1e-5, 1e-2, 20).tolist() #20
learning_rates = xp.linspace(1e-4, 1e-2, 20).tolist() #20
drop_rates = (np.arange(15)*0.05).tolist() # 15

class SpamDetector(nn.Module):
    def __init__(self, dropouts, input_size):
        super(SpamDetector, self).__init__()
        self.M = len(dropouts)
        self.weights = nn.ParameterList([torch.empty(input_size, 1) for _ in range(self.M)])
        self.biases = nn.ParameterList([torch.empty(1) for _ in range(self.M)])
        self.register_buffer("dropouts", torch.Tensor(dropouts))
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
            x_batched = x_batched * mask / keep_prob
        weights = torch.stack([w for w in self.weights], dim=0).view(self.M, F, 1).to(x.device)
        biases = torch.stack([b for b in self.biases], dim=0).view(self.M, 1, 1).to(x.device)
        out = torch.matmul(x_batched, weights) + biases
        out = out.squeeze()
        return out


# shape = (batch_sizes, epochs, optimizers, weighted_decays, learning_rates, drop_rates)
combos = []
for wd in weighted_decays:
    for lr in learning_rates:
        for dr in drop_rates:
            combos.append((wd, lr, dr))
M = len(combos)
DR_flat = [dr for (_, _, dr) in combos]
base_model = SpamDetector(DR_flat, X_tensor.shape[1])
init_state = copy.deepcopy(base_model.state_dict())
def create_param_groups(model, combos):
    param_groups = []
    for i, (wd_i, lr_i, dr_i) in enumerate(combos):
        param_groups.append({
            "params": [model.weights[i], model.biases[i]],
            "lr": float(lr_i),
            "weight_decay": float(wd_i)
        })

    return param_groups

optimizers = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD
}

info = []
print("beginning")
#t1 = time.time()
for i, loader in enumerate(loaders):
    load_info = []
    for j, (name, optim_cls) in enumerate(optimizers.items()):
        optim_info = []
        print(3*i+j, "/", 12)
        model1 = SpamDetector(DR_flat, X_tensor.shape[1])
        model1.load_state_dict(init_state)
        model1 = model1.to(device)
        param_groups = create_param_groups(model1, combos)
        optim = optim_cls(param_groups)
        for epoch in range(20):
            t1 = time.time()
            epoch_info = []
            model1.train()
            torch.cuda.synchronize()
            print("Num Batches", len(loader))
            for i, (X_batch, y_batch) in enumerate(loader):
                print("Batch: ", i)
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                y_pred = model1(X_batch).squeeze()
                y_batch = y_batch.expand(y_pred.shape[0], -1)
                loss = criterion(y_pred, y_batch)
                optim.zero_grad()
                loss.backward()
                optim.step()
            torch.cuda.synchronize()
            t2 = time.time()
            print("Training Time: ", t2-t1)
            model1.eval()
            with torch.no_grad():
                for X_batch, y_batch in eval_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    y_pred = model1(X_batch).squeeze()
                    y_batch = y_batch.expand(y_pred.shape[0], -1)
                    acc = (y_pred > 0.0).eq(y_batch).float().mean(dim=-1)
                    epoch_info.append(acc.cpu())
            t3 = time.time()
            print("Evaluation Time: ", t3-t2)
            optim_info.append(epoch_info)
        load_info.append(optim_info)
    info.append(load_info)
torch.cuda.synchronize()
#t2 = time.time()
#print("Time: ", t2-t1)
data = np.asarray(info)
data = data.squeeze()
shape = data.shape
data = data.reshape(shape[0], shape[1], shape[2], len(weighted_decays), len(learning_rates), len(drop_rates))
np.save("hyperparameter_data.npy", data)