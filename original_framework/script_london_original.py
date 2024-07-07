import pandas as pd
import torch as th
from tabula import Tabula

data = pd.read_csv("../Real_Datasets/London_original.csv")

data = data.iloc[0:3000]

model = Tabula(llm='distilgpt2', experiment_dir = "original_framework", batch_size=15, epochs=200)

model.fit(data)

th.save(model.model.state_dict(), "/u/dssc/eruoppolo/project/model_London_original200epoch.pt")

prompts = ["date " + str(row['date']) for _, row in data.iterrows()]

synthetic_data = model.tabula_sample(starting_prompts=prompts)

synthetic_data.to_csv("synt_datasets/London_Weather_original_fromprompts.csv", index=False)

print('batch_size=15, epochs=200, data = London_original.csv, output = London_Weather_original_fromprompts.csv')
