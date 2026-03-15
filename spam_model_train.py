import pandas as pd
import numpy as np
import cupy as cp # Delete if CuPy isn't available
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
spam_rate = np.sum(condition)/rows # Rate of messages that are spam
non_spam_rate = np.sum(~condition)/rows # Rate of messages that are not spam
print("Spam Data: ", np.sum(condition)) # Less spam data than non-spam, make "spam" messages weigh more
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

# Vectorizers
char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2) # Split into small sequences of characters
word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2) # Split into 1-3 word phrases
sent_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(4, 8), min_df=2) # Split into 4-8 word phrases or sentences

# Apply TF-IDF using the vectorizers
X_char = char_vectorizer.fit_transform(x_train)
X_word = word_vectorizer.fit_transform(x_train)
X_sent = sent_vectorizer.fit_transform(x_train)

# Get specific metrics using engineered_features function from spam_model_arch.py
X_engineered = np.array([engineered_features(m) for m in x_train]).astype(np.float32)
scaler = StandardScaler()
X_engineered = scaler.fit_transform(X_engineered)

# Turn into PyTorch tensors
X_char_tensor = torch.tensor(X_char.toarray(), dtype=torch.float32)
X_word_tensor = torch.tensor(X_word.toarray(), dtype=torch.float32)
X_sent_tensor = torch.tensor(X_sent.toarray(), dtype=torch.float32)
X_engineered_tensor = torch.tensor(X_engineered, dtype=torch.float32)
y_tensor = torch.tensor((y_train=="spam"), dtype=torch.float32)

# Device that the model will train on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create datasets
dataset_char = TensorDataset(X_char_tensor, y_tensor)
dataset_word = TensorDataset(X_word_tensor, y_tensor)
dataset_sent = TensorDataset(X_sent_tensor, y_tensor)
dataset_engineered = TensorDataset(X_engineered_tensor, y_tensor)

# Create Loaders
loader_char = DataLoader(dataset_char, batch_size=64, shuffle=True)
loader_word = DataLoader(dataset_word, batch_size=64, shuffle=True)
loader_sent = DataLoader(dataset_sent, batch_size=64, shuffle=True)
loader_engineered = DataLoader(dataset_engineered, batch_size=64, shuffle=True)

# Create models with features as input
model_char = SpamDetector(input_size=X_char_tensor.shape[1])
model_word = SpamDetector(input_size=X_word_tensor.shape[1])
model_sent = SpamDetector(input_size=X_sent_tensor.shape[1])
model_engineered = SpamDetector(input_size=X_engineered_tensor.shape[1])

# Binary Cross Entropy for binary classification
# Additionally make the weight for "spam" messages higher based on the ratio of non spam to spam
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([non_spam_rate/spam_rate]))

# Create optimizers for each model
optimizer_char = torch.optim.Adam(model_char.parameters(), lr=0.001, weight_decay=0)
optimizer_word = torch.optim.Adam(model_word.parameters(), lr=0.001, weight_decay=0)
optimizer_sent = torch.optim.Adam(model_sent.parameters(), lr=0.001, weight_decay=0)
optimizer_engineered = torch.optim.Adam(model_engineered.parameters(), lr=0.001, weight_decay=0)

criterion = criterion.to(device) # Send criterion to device

# Create lists to iterate over
models = [model_char, model_word, model_sent, model_engineered]
optimizers = [optimizer_char, optimizer_word, optimizer_sent, optimizer_engineered]
loaders = [loader_char, loader_word, loader_sent, loader_engineered]

start = time.time() # Starting time
for i, (model, optimizer, loader) in enumerate(zip(models, optimizers, loaders)):
    print("Model: ", i, "\n")
    model = model.to(device) # Send model to device
    model.train() # Allow weights and biases to be adjusted

    if i == 3: # Model that trains on engineered features takes more time to learn the correlations
        epoch_num = 20
    else:
        epoch_num = 10

    for epoch in range(epoch_num):
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
X_sent = sent_vectorizer.transform(x_eval)
X_engineered = np.array([engineered_features(m) for m in x_eval]).astype(np.float32)
X_enginnered = scaler.transform(X_engineered)

# Turn into PyTorch Tensors
X_char_tensor = torch.tensor(X_char.toarray(), dtype=torch.float32).to(device)
X_word_tensor = torch.tensor(X_word.toarray(), dtype=torch.float32).to(device)
X_sent_tensor = torch.tensor(X_sent.toarray(), dtype=torch.float32).to(device)
X_enginnered_tensor = torch.tensor(X_enginnered, dtype=torch.float32).to(device)
y_tensor = torch.tensor((y_eval=="spam"), dtype=torch.float32).to(device)

# Freeze weights and biases so they can't be adjusted
model_char.eval()
model_word.eval()
model_sent.eval()
model_engineered.eval()

# Predictions of each model
char_pred = model_char(X_char_tensor).squeeze()
word_pred = model_word(X_word_tensor).squeeze()
sent_pred = model_sent(X_sent_tensor).squeeze()
engineered_pred = model_engineered(X_enginnered_tensor).squeeze()

# Sum the logits to get the overall prediction
pred = char_pred + word_pred + sent_pred + engineered_pred

# Get average accuracy
acc = 100*(pred > 0.0).eq(y_tensor).float().mean()

# Print information
print("Accuracy: ", acc.item())
print("Avg Logits: ", pred.mean().item())
print("STD Logits: ", pred.std().item())
print("Max Logits: ", pred.max().item())
print("Min Logits: ", pred.min().item())

# Save the models
torch.save(model_char, "models/char_model.pth")
torch.save(model_word, "models/word_model.pth")
torch.save(model_sent, "models/sent_model.pth")
torch.save(model_engineered, "models/engineered_model.pth")

# Save the vectorizers and the scaler
joblib.dump(char_vectorizer, "vect_and_scale/char_vectorizer.joblib")
joblib.dump(word_vectorizer, "vect_and_scale/word_vectorizer.joblib")
joblib.dump(sent_vectorizer, "vect_and_scale/sent_vectorizer.joblib")
joblib.dump(scaler, "vect_and_scale/scaler.joblib")