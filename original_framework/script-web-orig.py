import pandas as pd
import torch as th
from tabula import Tabula

data = pd.read_csv("../Real_Datasets/web_orig.csv")
model = Tabula(llm='distilgpt2', experiment_dir = "original_framework", batch_size=15, epochs=200)

model.fit(data)
th.save(model.model.state_dict(), "/u/dssc/eruoppolo/project/original_web200epoch.pt")

prompts = ["Date " + str(row['Date']) for _, row in data.iterrows()]

synthetic_data = model.tabula_sample(starting_prompts=prompts)
synthetic_data.to_csv("synt_datasets/original_web_fromprompts.csv", index=False)

print('epochs:120, batch size: 16, data = web_orig.csv,  description: generating on web raw data as baseline')
