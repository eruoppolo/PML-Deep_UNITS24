import pandas as pd
import torch as th
from tabula import Tabula

data = pd.read_csv("../Real_Datasets/complete_dataset_n.csv")
model = Tabula(llm='distilgpt2', experiment_dir = "original_framework", batch_size=15, epochs=200)

model.fit(data)

th.save(model.model.state_dict(), "/u/dssc/eruoppolo/project/energy_original_200ep.pt")

prompts = ["date "+ str(row['date']) for _, row in data.iterrows()]

synthetic_data = model.tabula_sample(starting_prompts=prompts)

synthetic_data.to_csv("synt_datasets/original_energy.csv", index=False)

print("data = complete_dataset_n.csv;    batch_size=15, epochs=200")
