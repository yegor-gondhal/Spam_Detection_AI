import pandas as pd
import numpy as np
#import cupy as cp

xp = np

data = pd.read_csv("spam.csv")
print("Categories: ", data.columns)
print("Rows: ", data.shape[0])