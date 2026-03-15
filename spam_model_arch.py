import torch
import torch.nn as nn
import re
import numpy as np

# Define Model
class SpamDetector(nn.Module):
    def __init__(self, input_size):
        super(SpamDetector, self).__init__()
        self.linear = nn.Linear(input_size, 128)
        self.linear1 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.linear(x) # Put input data through the neural layers
        x = self.linear1(x)
        return x

# Function for determining specific features of a message
def engineered_features(message):
    has_phone = int(bool(re.search(r"\d{3}[- ]?\d{3}[- ]?\d{4}", message))) # Search for phone numbers
    has_url = int(bool(re.search(r"https|www.", message))) # Search for website links
    num_exclaim = message.count("!") # Search for number of exclamation points
    num_digits = sum(c.isdigit() for c in message) # Search for number of digits in the message
    num_dollar = message.count("$") # Search for number of money symbols
    length = len(message) # Take into account the length of the message

    return np.array([has_phone, has_url, num_exclaim, num_digits, num_dollar, length]) # Return an array of this information