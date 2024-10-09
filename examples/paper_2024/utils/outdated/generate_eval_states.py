import numpy as np
import pandas as pd
from model import sample_state_space

np.random.seed(0)

num_ICs = 10

# Generate ICs
data = sample_state_space(num_ICs)
data = np.hstack(data).T

# Specify the CSV file path
csv_file_path = "ICs.csv"
df = pd.DataFrame(data)
df.to_csv(csv_file_path, header=False, index=False)
print(f"CSV file '{csv_file_path}' has been generated.")
