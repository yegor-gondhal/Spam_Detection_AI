import torch
import torch.nn as nn
import re
import numpy as np

# Define Model
class SpamDetector(nn.Module):
    def __init__(self, input_size):
        super(SpamDetector, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.linear(x) # Put input through linear layer
        #x = self.sigmoid(x) # Apply sigmoid function
        return x

def bayes_prior_correction(pred):
    training_rate = 0.5
    real_rate = 747/5572
    x = (1-training_rate)*(1-pred)/(1-real_rate)
    x += training_rate*pred/real_rate
    p = training_rate*pred/real_rate
    return p/x

def engineered_features(message):
    has_phone = int(bool(re.search(r"\d{3}[- ]?\d{3}[- ]?\d{4}", message)))
    has_url = int(bool(re.search(r"https|www.", message)))
    num_exclaim = message.count("!")
    num_digits = sum(c.isdigit() for c in message)
    num_dollar = message.count("$")
    length = len(message)

    return np.array([has_phone, has_url, num_exclaim, num_digits, num_dollar, length])