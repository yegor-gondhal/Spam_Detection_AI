import pandas as pd
import numpy as np
import cupy as cp # Delete if CuPy isn't available
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
from spam_model_arch import SpamDetector, engineered_features

xp = cp # Change to np if CuPy isn't available

# Load Data
data = pd.read_csv("spam.csv", encoding="latin1", usecols=[0, 1])

# Make sure there are no empty cells
if (np.sum(data.isna()) != 0):
    empty_rows = data.isna().any(axis=1)
    raise ValueError("Empty Cells Detected")

# Info about the data
rows = data.shape[0]
print("Rows: ", rows)
condition = data["v1"] == "spam"
spam_rate = np.sum(condition)/rows
non_spam_rate = np.sum(~condition)/rows
print("Spam Data: ", np.sum(condition)) # Less spam data than non-spam, use SMOTE for imbalanced datasets
print("Nonspam Data: ", np.sum(~condition), "\n")

data = data.to_numpy()
data = np.asarray(data)

num_training_rows = round(rows * 0.9) # 90% training, 10% testing
# Index data
training_data = data[:num_training_rows, :]
evaluation_data = data[num_training_rows:, :]
# Assign x and y
x_train, y_train = training_data[:, 1], training_data[:, 0]
x_eval, y_eval = evaluation_data[:, 1], evaluation_data[:, 0]

smote = SMOTE(random_state=42)

char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3,5))
word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 3))
X_char = char_vectorizer.fit_transform(x_train) # Apply TF-IDF
X_word = word_vectorizer.fit_transform(x_train)
X_engineered = np.array([engineered_features(m) for m in x_train]).astype(np.float32)
scaler = StandardScaler()
X_engineered = scaler.fit_transform(X_engineered)

# Turn into PyTorch tensors
X_char_tensor = torch.tensor(X_char.toarray(), dtype=torch.float32)
X_word_tensor = torch.tensor(X_word.toarray(), dtype=torch.float32)
X_engineered_tensor = torch.tensor(X_engineered, dtype=torch.float32)
y_tensor = torch.tensor((y_train=="spam"), dtype=torch.float32)

# Device that the model will train on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset and loader
dataset_char = TensorDataset(X_char_tensor, y_tensor)
dataset_word = TensorDataset(X_word_tensor, y_tensor)
dataset_engineered = TensorDataset(X_engineered_tensor, y_tensor)
loader_char = DataLoader(dataset_char, batch_size=64, shuffle=True)
loader_word = DataLoader(dataset_word, batch_size=64, shuffle=True)
loader_engineered = DataLoader(dataset_engineered, batch_size=64, shuffle=True)

# Model with features as input
model_char = SpamDetector(input_size=X_char_tensor.shape[1])
model_word = SpamDetector(input_size=X_word_tensor.shape[1])
model_engineered = SpamDetector(input_size=X_engineered_tensor.shape[1])

# Binary Cross Entropy for binary classification
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([non_spam_rate/spam_rate]))
optimizer_char = torch.optim.Adam(model_char.parameters(), lr=0.001, weight_decay=0)
optimizer_word = torch.optim.Adam(model_word.parameters(), lr=0.001, weight_decay=0)
optimizer_engineered = torch.optim.Adam(model_engineered.parameters(), lr=0.001, weight_decay=0)

criterion = criterion.to(device) # Send criterion to device
models = [model_char, model_word, model_engineered]
optimizers = [optimizer_char, optimizer_word, optimizer_engineered]
loaders = [loader_char, loader_word, loader_engineered]
start = time.time() # Starting time
for i, (model, optimizer, loader) in enumerate(zip(models, optimizers, loaders)):
    print("Model: ", i, "\n")
    model = model.to(device) # Send model to device
    model.train() # Allow weights and biases to be adjusted
    for epoch in range(10): # 3 epochs of training
        for X_batch, y_batch in loader: # Iterate through every batch in the loader
            # Send to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            # Receive output from model
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch) # Find the error
            optimizer.zero_grad() # Set gradients from previous batches to 0 (only let current loop affect model)
            loss.backward() # Back-propagation
            optimizer.step() # Adjust weights and biases

        # Info
        print("Epoch: ", epoch)
        print("Loss: ", loss.item(), "\n")

end = time.time() # Ending time
print("Elapsed Time: ", end - start)

# Begin creating evaluation dataset
X_char = char_vectorizer.transform(x_eval)
X_word = word_vectorizer.transform(x_eval) # Transform using TF-IDF
X_engineered = np.array([engineered_features(m) for m in x_eval]).astype(np.float32)
X_enginnered = scaler.transform(X_engineered)
# Turn into PyTorch Tensors
X_char_tensor = torch.tensor(X_char.toarray(), dtype=torch.float32).to(device)
X_word_tensor = torch.tensor(X_word.toarray(), dtype=torch.float32).to(device)
X_enginnered_tensor = torch.tensor(X_enginnered, dtype=torch.float32).to(device)
y_tensor = torch.tensor((y_eval=="spam"), dtype=torch.float32).to(device)


model_char.eval()
model_word.eval()
model_engineered.eval()

char_pred = model_char(X_char_tensor).squeeze()
word_pred = model_word(X_word_tensor).squeeze()
engineered_pred = model_engineered(X_enginnered_tensor).squeeze()


pred = char_pred + word_pred + engineered_pred


acc = 100*(pred > 0.0).eq(y_tensor).float().mean()
print("Accuracy: ", acc.item())
print("Avg Logits: ", pred.mean().item())
print("STD Logits: ", pred.std().item())
print("Max Logits: ", pred.max().item())
print("Min Logits: ", pred.min().item())


torch.save(model_char, "models/char_model.pth")
torch.save(model_word, "models/word_model.pth")
torch.save(model_engineered, "models/engineered_model.pth")

joblib.dump(char_vectorizer, "vect_and_scale/char_vectorizer.joblib")
joblib.dump(word_vectorizer, "vect_and_scale/word_vectorizer.joblib")
joblib.dump(scaler, "vect_and_scale/scaler.joblib")