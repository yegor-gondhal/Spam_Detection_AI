import pandas as pd
import numpy as np
#import cupy as cp
from imblearn.over_sampling import SMOTE

xp = np

data = pd.read_csv("spam.csv", encoding="latin1", usecols=[0, 1])


if (xp.sum(data.isna()) != 0):
    empty_rows = data.isna().any(axis=1)
    raise ValueError("Empty Cells Detected")

rows = data.shape[0]
print("Rows: ", rows)
condition = data["v1"] == "spam"
print("Spam Data: ", xp.sum(condition))
print("Nonspam Data: ", xp.sum(~condition))

num_training_rows = round(rows * 0.9)
training_data = data.iloc[:num_training_rows]
evaluation_data = data.iloc[num_training_rows:]
print(type(training_data["v1"]))
smote = SMOTE(random_state=42)
print(xp.size(training_data["v1"]))
print(training_data["v1"].shape)
training_data = smote.fit_resample(training_data["v1"], training_data["v2"])
print(xp.size(training_data["v1"]))