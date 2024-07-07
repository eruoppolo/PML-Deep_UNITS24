import pandas as pd
import torch as th
from tabula import Tabula

print('\n======== weather dataset =======\n')

data = pd.read_csv("../Real_Datasets/London_complete.csv")

data = data.iloc[0:8000]

model = Tabula(llm='distilgpt2', experiment_dir = "time_series_framework", batch_size=16, epochs=30, seq_len=5, stride=2)

model.fit(data)

th.save(model.model.state_dict(), "/u/dssc/eruoppolo/project/model_ft_weath_windowed30.pt")

promp_data = data.iloc[0:2500]
prompts = ["day " + str(int(row['day'])) + ", month " + str(int(row['month'])) + ", year " + str(int(row['year'])) for _, row in promp_data.iterrows()]

synthetic_data = model.tabula_sample(starting_prompts=prompts)

synthetic_data.to_csv("window_datasets/weat_window.csv", index=False)

print(f'epochs:30, batch size: 16, seq_len=5, stride=2,  data = London_complete.csv,  out: weat_window.csv,    description: generating after training on weather')

print('\n======== energy dataset =======\n')

data = pd.read_csv("../Real_Datasets/energy_prep_c.csv")

model = Tabula(llm='distilgpt2', experiment_dir = "time_series_framework", batch_size=16, epochs=50, seq_len=5, stride=2)

model.model.load_state_dict(th.load("/u/dssc/eruoppolo/project/model_ft_weath_windowed30.pt"), strict=False)

model.fit(data)

th.save(model.model.state_dict(), "/u/dssc/eruoppolo/project/model_energy_windowed50.pt")

prompts = ["MonthDay " + str(row['MonthDay']) + ", Month " + str(row['Month']) + ", Year " + str(row['Year']) for _, row in data.iterrows()]

synthetic_data = model.tabula_sample(starting_prompts=prompts)

synthetic_data.to_csv("window_datasets/energy_window.csv", index=False)

print(f'epochs:50, batch size: 16, seq_len=5, stride=2, out: energy_window.csv, data = energy_prep_c.csv, description: generating from weather model after finetuning on energy')

print('\n======== web dataset =======\n')

data = pd.read_csv("../Real_Datasets/web_prep.csv")

model = Tabula(llm='distilgpt2', experiment_dir = "time_series_framework", batch_size=16, epochs=50, seq_len=7, stride=2)

model.model.load_state_dict(th.load("/u/dssc/eruoppolo/project/model_energy_windowed50.pt"), strict=False)

model.fit(data)

th.save(model.model.state_dict(), "/u/dssc/eruoppolo/project/model_ft_web_windowed50.pt")

prompts = ["MonthDay " + str(int(row['MonthDay'])) + ", Month " + str(int(row['Month'])) + ", Year " + str(int(row['Year'])) for _, row in data.iterrows()]

synthetic_data = model.tabula_sample(starting_prompts=prompts)

synthetic_data.to_csv("window_datasets/web_window.csv", index=False)

print(f'epochs:50, batch size: 16, seq_len=7, stride=2,  data = web_prep.csv,  out: web_window.csv,    description: generating web after fine tuning energy model already finetuned on weather')









