import torch
import joblib
from spam_model_arch import bayes_prior_correction
import numpy as np

message = np.array(["W1N 25 DOLLAR$ NOW!!! Txt 801-578-8375 fr more info."])
device = torch.device("cpu")
vectorizer = joblib.load("spam_vectorizer.joblib")
X = vectorizer.transform(message)
X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
model = torch.load("spam_model.pth", weights_only=False).to(device)
pred = model(X_tensor)

print("Message: ", message[0])

if pred > 0:
    print("Category: Spam")
else:
    print("Category: Not Spam")
pred = torch.sigmoid(pred).item()
print("Confidence: ", round(100*pred, 2))
#print("Confidence after Bayes: ", round(100*bayes_prior_correction(pred), 2))