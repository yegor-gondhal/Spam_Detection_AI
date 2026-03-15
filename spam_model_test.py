import torch
import joblib
import numpy as np
from spam_model_arch import engineered_features

#message = "W1N 25 DOLLAR$ NOW!!! Txt 801-578-8375 fr more info."
#message = "OMGGG!! I JUST GOT HIS NUMBER, ITS 801-578-8375"
message = "Hey Susan!, You have 5 valid credits available! Click here to use them now!"
device = torch.device("cpu")

char_vectorizer = joblib.load("vect_and_scale/char_vectorizer.joblib")
word_vectorizer = joblib.load("vect_and_scale/word_vectorizer.joblib")
sent_vectorizer = joblib.load("vect_and_scale/sent_vectorizer.joblib")
scaler = joblib.load("vect_and_scale/scaler.joblib")

X_char = char_vectorizer.transform([message])
X_word = word_vectorizer.transform([message])
X_sent = sent_vectorizer.transform([message])
X_engineered = engineered_features(message)
X_engineered = scaler.transform(X_engineered.reshape(1, -1))

X_char_tensor = torch.tensor(X_char.toarray(), dtype=torch.float32)
X_word_tensor = torch.tensor(X_word.toarray(), dtype=torch.float32)
X_sent_tensor = torch.tensor(X_sent.toarray(), dtype=torch.float32)
X_engineered_tensor = torch.tensor(X_engineered, dtype=torch.float32)

char_model = torch.load("models/char_model.pth", weights_only=False).to(device)
word_model = torch.load("models/word_model.pth", weights_only=False).to(device)
sent_model = torch.load("models/sent_model.pth", weights_only=False).to(device)
engineered_model = torch.load("models/engineered_model.pth", weights_only=False).to(device)

char_pred = char_model(X_char_tensor)
word_pred = word_model(X_word_tensor)
sent_pred = sent_model(X_sent_tensor)
engineered_pred = engineered_model(X_engineered_tensor)
pred = char_pred + word_pred + sent_pred + engineered_pred

print("Message: ", message)

if pred > 0:
    print("Category: Spam")
else:
    print("Category: Not Spam")

pred = torch.sigmoid(pred).item()
print("Confidence Percentage: ", round(100*pred, 2))